from .ir import (
    MathNode, var, const, add, matmul, affine, grad, constraint, leq,
    op_class, COMMUTATIVE_OPS,
)
from .parser import parse, ParseError
from .normalize import normalize
from .embedder import MathEmbedder, cosine_similarity, pairwise_cosine, EMBED_DIM
from .topology import (
    TopologyBuilder, MaskDiagnostics,
    symbolic_dependency_matrix, same_operator_matrix,
    embedding_topk_matrix, local_window_matrix, identity_matrix,
    shape_compatibility_matrix,
)
from .shape import infer_shape, infer_tree, ShapeError
from .attention import DenseMaskedMathAttention, MathRoutedAttention, FullAttention, math_attention
from .sparse_attention import neighbor_attention, neighbors_from_mask, max_k_from_mask
from .router import OperatorRouter, RouteResult, EXPERT_NAMES
from .verifier import Verifier, ExecutionPlan, VerificationResult
from .tasks import EXPERT_TO_ID, ID_TO_EXPERT, N_EXPERTS

__all__ = [
    # ir
    "MathNode", "var", "const", "add", "matmul", "affine", "grad",
    "constraint", "leq", "op_class", "COMMUTATIVE_OPS",
    # parser
    "parse", "ParseError",
    # normalize
    "normalize",
    # embedder
    "MathEmbedder", "cosine_similarity", "pairwise_cosine", "EMBED_DIM",
    # topology
    "TopologyBuilder", "MaskDiagnostics",
    "symbolic_dependency_matrix", "same_operator_matrix",
    "embedding_topk_matrix", "local_window_matrix", "identity_matrix",
    "shape_compatibility_matrix",
    # shape
    "infer_shape", "infer_tree", "ShapeError",
    # attention
    "DenseMaskedMathAttention", "MathRoutedAttention", "FullAttention", "math_attention",
    # sparse attention
    "neighbor_attention", "neighbors_from_mask", "max_k_from_mask",
    # router
    "OperatorRouter", "RouteResult", "EXPERT_NAMES",
    # verifier
    "Verifier", "ExecutionPlan", "VerificationResult",
    # tasks
    "EXPERT_TO_ID", "ID_TO_EXPERT", "N_EXPERTS",
]
