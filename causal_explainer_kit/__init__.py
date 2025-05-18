# causal_explainer_kit/__init__.py

# Import key functions and classes to make them available at the top-level of the package
from .data_processing import load_uci_repository, generate_random_row, feature_engineering_uci_repository
from .modeling import train_model
from .explainers import CausalExplainer
from .visualization import display_causal_graph_customized

# You can also define __version__ here
__version__ = "0.1.0"

__all__ = [
    "load_uci_repository",
    "generate_random_row",
    "feature_engineering_uci_repository", # If you want to expose it
    "train_model",
    "CausalExplainer",
    "display_causal_graph_customized",
    "__version__"
]