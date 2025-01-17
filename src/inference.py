from typing import List, Optional
import json
import os
import faiss  # @manual=//faiss/python:pyfaiss_gpu
import faiss.contrib.torch_utils  # @manual=//faiss/contrib:faiss_contrib_gpu
import torch

from torchrec.datasets.movielens import DEFAULT_RATINGS_COLUMN_NAMES
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.modules.embedding_configs import DataType, EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.metadata import ShardMetadata 
from torch.distributed.remote_device import _remote_device
from torchrec.distributed.types import ShardingType
from knn_index import get_index
from torch import distributed as dist

# pyre-ignore[21]
# @manual=//torchrec/github/examples/retrieval/modules:two_tower
from modules.two_tower import (  # noqa F811
    convert_TwoTower_to_TwoTowerRetrieval,
    TwoTowerRetrieval,
)

two_tower_column_names = DEFAULT_RATINGS_COLUMN_NAMES[:2]
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
rank = int(os.environ.get("RANK", "0"))

def load_model(num_embeddings: int = 1024 * 1024,
            embedding_dim: int = 64,
            layer_sizes: Optional[List[int]] = None,
            num_centroids: int = 100,
            k: int = 100,
            num_subquantizers: int = 8,
            bits_per_code: int = 8,
            num_probe: int = 8,
            model_device_idx: int = 0,
            faiss_device_idx: int = 0,
            batch_size: int = 32,
            load_dir: Optional[str] = "/opt/ml/model",
            world_size: int = 1,
        ) -> DistributedModelParallel:
        """
        Loads the serialized model and FAISS index from `two_tower_train.py`.
        A `TwoTowerRetrieval` model is instantiated, which wraps the `KNNIndex`, the query (user) tower and the candidate item (movie) tower inside an `nn.Module`.
        The retreival model is quantized using [`torchrec.quant`](https://pytorch.org/torchrec/torchrec.quant.html).
        The serialized `TwoTower` model weights trained before are converted into `TwoTowerRetrieval` which are loaded into the retrieval model.
        The seralized trained FAISS index is also loaded.
        The entire retreival model can be queried with a batch of candidate (user) ids and returns logits which can be used in ranking.

        Args:
            num_embeddings (int): The number of embeddings the embedding table
            embedding_dim (int): embedding dimension of both embedding tables
            layer_sizes (str): Comma separated list representing layer sizes of the MLP. Last size is the final embedding size
            num_centroids (int): The number of centroids (Voronoi cells)
            k (int): The number of nearest neighbors to retrieve
            num_subquantizers (int): The number of subquanitizers in Product Quantization (PQ) compression of subvectors
            bits_per_code (int): The number of bits for each subvector in Product Quantization (PQ)
            num_probe (int): The number of centroids (Voronoi cells) to probe. Must be <= num_centroids. Sweeping powers of 2 for nprobe and picking one of those based on recall statistics (e.g., 1, 2, 4, 8, ..,) is typically done.
            model_device_idx (int): device index to place model on
            faiss_device_idx (int): device index to place FAISS index on
            batch_size (int): batch_size of the random batch used to query Retrieval model at the end of the script
            load_dir (Optional[str]): Directory to load model and faiss index from. If None, uses random data
        """
        if layer_sizes is None:
            layer_sizes = [128, 64]
        assert torch.cuda.is_available(), "This example requires a GPU"

        device: torch.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        two_tower_column_names = DEFAULT_RATINGS_COLUMN_NAMES[:2]
        ebcs = []
        for feature_name in two_tower_column_names:
            config = EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                feature_names=[feature_name],
                data_type=DataType.FP16,
            )
            ebcs.append(
                EmbeddingBagCollection(
                    tables=[config],
                    device=torch.device("meta"),
                )
            )

        retrieval_sd = None
        if load_dir is not None:
            load_dir = load_dir.rstrip("/")
            # pyre-ignore[16]
            index = faiss.index_cpu_to_gpu(
                # pyre-ignore[16]
                faiss.StandardGpuResources(),
                faiss_device_idx,
                # pyre-ignore[16]
                faiss.read_index(f"{load_dir}/faiss.index"),
            )
            backend = "nccl"
            dist.init_process_group(backend=backend, world_size=world_size, rank=local_rank)
            with torch.serialization.safe_globals([ShardedTensor, Shard, ShardMetadata, _remote_device]):
                two_tower_sd = torch.load(f"{load_dir}/model.pt", weights_only=False)
            retrieval_sd = convert_TwoTower_to_TwoTowerRetrieval(
                two_tower_sd,
                [f"t_{two_tower_column_names[0]}"],
                [f"t_{two_tower_column_names[1]}"],
            )
        else:
            embeddings = torch.rand((num_embeddings, embedding_dim)).to(
                torch.device(f"cuda:{faiss_device_idx}")
            )
            index = get_index(
                embedding_dim=embedding_dim,
                num_centroids=num_centroids,
                num_probe=num_probe,
                num_subquantizers=num_subquantizers,
                bits_per_code=bits_per_code,
                device=torch.device(f"cuda:{faiss_device_idx}"),
            )
            index.train(embeddings)
            index.add(embeddings)

        retrieval_model = TwoTowerRetrieval(
            index, ebcs[0], ebcs[1], layer_sizes, k, device)

        constraints = {}
        for feature_name in two_tower_column_names:
            constraints[f"t_{feature_name}"] = ParameterConstraints(
                sharding_types=[ShardingType.TABLE_WISE.value],
                compute_kernels=[EmbeddingComputeKernel.QUANT.value],
            )

        dmp = DistributedModelParallel(
            module=retrieval_model,
            device=device,
            # env=ShardingEnv.from_local(world_size=world_size, rank=model_device_idx),
            init_data_parallel=False,
        )

        if retrieval_sd is not None:
            dmp.load_state_dict(retrieval_sd)

        return dmp

def model_fn(model_dir):
    """Load the model for inference"""
    model = load_model()
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    """Deserialize and prepare the prediction input"""
    if request_content_type == "application/json":
        input_data = json.loads(request_body)["inputs"]
        return torch.tensor(input_data, dtype=torch.int64)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Perform prediction"""

    device = torch.device(f"cuda:{rank}")
    batch_size = input_data.shape[0]
    batch = KeyedJaggedTensor(
        keys=[two_tower_column_names[0]],
        values=input_data.to(device),
        lengths=torch.tensor([1] * batch_size, device=device),
    )

    actual_result = model(batch)
    # Convert to NumPy array
    numpy_data = actual_result.detach().cpu().numpy()
    return numpy_data

def output_fn(prediction, response_content_type):
    """Serialize and return the prediction output"""
    if response_content_type == "application/json":
        response = prediction.tolist()
        return json.dumps({"predictions": response})
    raise ValueError(f"Unsupported content type: {response_content_type}")
