import argparse
import os
import shutil
import glob
from typing import List, Optional

import click

import faiss  # @manual=//faiss/python:pyfaiss_gpu
import faiss.contrib.torch_utils  # @manual=//faiss/contrib:faiss_contrib_gpu
import torch
from torch import distributed as dist
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec import inference as trec_infer
from torchrec.datasets.movielens import DEFAULT_RATINGS_COLUMN_NAMES
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.inference.state_dict_transform import (
    state_dict_gather,
    state_dict_to_device,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torch.distributed._sharded_tensor import ShardedTensor
import mlflow
from time import gmtime, strftime, sleep
from data.dataloader import get_dataloader

# pyre-ignore[21]
# @manual=//torchrec/github/examples/retrieval:knn_index
from knn_index import get_index

# pyre-ignore[21]
# @manual=//torchrec/github/examples/retrieval/modules:two_tower
from modules.two_tower import TwoTower, TwoTowerTrainTask



def train(
    num_embeddings: int = 1024**2,
    embedding_dim: int = 64,
    layer_sizes: Optional[List[int]] = None,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    num_iterations: int = 100,
    num_centroids: int = 100,
    num_subquantizers: int = 8,
    bits_per_code: int = 8,
    num_probe: int = 8,
    save_dir: Optional[str] = None,
    train_dir: str = None,
    mlflow_tracking_server_arn: Optional[str] = None,
    mlflow_experiment_name: Optional[str] = None
) -> None:
    """
    Trains a simple Two Tower (UV) model, which is a simplified version of [A Dual Augmented Two-tower Model for Online Large-scale Recommendation](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_4.pdf).
    Torchrec is used to shard the model, and is pipelined so that dataloading, data-parallel to model-parallel comms, and forward/backward are overlapped.
    It is trained on random data in the format of [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/) dataset in SPMD fashion.
    The distributed model is gathered to CPU.
    The item (movie) towers embeddings are used to train a FAISS [IVFPQ](https://github.com/facebookresearch/faiss/wiki/Lower-memory-footprint) index, which is serialized.
    The resulting `KNNIndex` can be queried with batched `torch.Tensor`, and will return the distances and indices for the approximate K nearest neighbors of the query embeddings. The model itself is also serialized.

    Args:
        num_embeddings (int): The number of embeddings the embedding table
        embedding_dim (int): embedding dimension of both embedding tables
        layer_sizes (List[int]): list representing layer sizes of the MLP. Last size is the final embedding size
        learning_rate (float): learning_rate
        batch_size (int): batch size to use for training
        num_iterations (int): number of train batches
        num_centroids (int): The number of centroids (Voronoi cells)
        num_subquantizers (int): The number of subquanitizers in Product Quantization (PQ) compression of subvectors
        bits_per_code (int): The number of bits for each subvector in Product Quantization (PQ)
        num_probe (int): The number of centroids (Voronoi cells) to probe. Must be <= num_centroids. Sweeping powers of 2 for nprobe and picking one of those based on recall statistics (e.g., 1, 2, 4, 8, ..,) is typically done.
        save_dir (Optional[str]): Directory to save model and faiss index. If None, nothing is saved
    """
    if layer_sizes is None:
        layer_sizes = [128, 64]

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"
    # dist.init_process_group(backend="gloo")
    dist.init_process_group(backend=backend, world_size=1, rank=0)

    two_tower_column_names = DEFAULT_RATINGS_COLUMN_NAMES[:2]
    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=[feature_name],
        )
        for feature_name in two_tower_column_names
    ]
    embedding_bag_collection = EmbeddingBagCollection(
        tables=eb_configs,
        device=torch.device("meta"),
    )
    two_tower_model = TwoTower(
        embedding_bag_collection=embedding_bag_collection,
        layer_sizes=layer_sizes,
        device=device,
    )
    two_tower_train_task = TwoTowerTrainTask(two_tower_model)
    apply_optimizer_in_backward(
        RowWiseAdagrad,
        two_tower_train_task.two_tower.ebc.parameters(),
        {"lr": learning_rate},
    )
    model = DistributedModelParallel(
        module=two_tower_train_task,
        device=device,
    )

    optimizer = KeyedOptimizerWrapper(
        dict(model.named_parameters()),
        lambda params: torch.optim.Adam(params, lr=learning_rate),
    )

    training_files = glob.glob(f"{train_dir}/*.*")
    dataloader = get_dataloader(
        batch_size=batch_size,
        num_embeddings=num_embeddings,
        pin_memory=(backend == "nccl"),
        csv_file=training_files[0]
    )

    dl_iterator = iter(dataloader)
    train_pipeline = TrainPipelineSparseDist(
        model,
        optimizer,
        device,
    )

     # Use MLFlow tracking server to capture metrics
    if mlflow_tracking_server_arn:
        mlflow.set_tracking_uri(mlflow_tracking_server_arn)
        mlflow.set_experiment(mlflow_experiment_name)
        suffix = strftime('%d-%H-%M-%S', gmtime())
        params = {}
        params['num_embeddings'] = num_embeddings
        params['layer_sizes'] = layer_sizes
        params['learning_rate'] = learning_rate
        params['batch_size'] = batch_size
        params['num_iterations'] = num_iterations
        params['num_centroids'] = num_centroids
        params['num_subquantizers'] = num_subquantizers
        params['bits_per_code'] = bits_per_code
        params['num_probe'] = num_probe
        with mlflow.start_run(run_name=f"two-tower-training-run-{suffix}") as run:
            mlflow.log_params(params)
            for step in range(num_iterations):
                try:
                    output = train_pipeline.progress(dl_iterator)
                    loss = output[0].to("cpu").item()
                    mlflow.log_metric("training_loss", loss, step=step)
                except StopIteration:
                    break
    else:
        for step in range(num_iterations):
            try:
                train_pipeline.progress(dl_iterator)
            except StopIteration:
                break

    checkpoint_pg = dist.new_group(backend="gloo")
    # Copy sharded state_dict to CPU.
    cpu_state_dict = state_dict_to_device(
        model.state_dict(), pg=checkpoint_pg, device=torch.device("cpu")
    )

    ebc_cpu = EmbeddingBagCollection(
        tables=eb_configs,
        device=torch.device("meta"),
    )
    two_tower_cpu = TwoTower(
        embedding_bag_collection=ebc_cpu,
        layer_sizes=layer_sizes,
    )
    two_tower_train_cpu = TwoTowerTrainTask(two_tower_cpu)
    if rank == 0:
        two_tower_train_cpu = two_tower_train_cpu.to_empty(device="cpu")
    state_dict_gather(cpu_state_dict, two_tower_train_cpu.state_dict())
    dist.barrier()

    # Create and train FAISS index for the item (movie) tower on CPU
    if rank == 0:
        index = get_index(
            embedding_dim=embedding_dim,
            num_centroids=num_centroids,
            num_probe=num_probe,
            num_subquantizers=num_subquantizers,
            bits_per_code=bits_per_code,
            device=torch.device("cpu"),
        )

        values = torch.tensor(list(range(num_embeddings)), device=torch.device("cpu"))
        kjt = KeyedJaggedTensor(
            keys=two_tower_column_names,
            values=values,
            lengths=torch.tensor(
                [0] * num_embeddings + [1] * num_embeddings,
                device=torch.device("cpu"),
            ),
        )

        # Get the embeddings of the item(movie) tower by querying model
        with torch.no_grad():
            lookups = two_tower_cpu.ebc(kjt)[two_tower_column_names[1]]
            item_embeddings = two_tower_cpu.candidate_proj(lookups)
        index.train(item_embeddings)
        index.add(item_embeddings)

        if save_dir is not None:
            save_dir = save_dir.rstrip("/")
            torch.save(model.state_dict(), f"{save_dir}/model.pt")
            # pyre-ignore[16]
            faiss.write_index(index, f"{save_dir}/faiss.index")


if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--use-cuda', type=bool, default=True)
    parser.add_argument('--num_embeddings', type=int, default=1024**2)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_centroids', type=int, default=100)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST', None))
    parser.add_argument('--mlflow-tracking-server-arn', type=str, default=None)
    parser.add_argument('--mlflow-experiment-name', type=str, default="two-tower-training")
    
    args, _ = parser.parse_known_args()

    train(num_embeddings=args.num_embeddings,
    embedding_dim=args.embedding_dim,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    num_iterations=args.epochs,
    num_centroids=args.num_centroids,
    save_dir=args.model_dir,
    train_dir=args.train,
    mlflow_tracking_server_arn=args.mlflow_tracking_server_arn,
    mlflow_experiment_name=args.mlflow_experiment_name)