# causal_explainer_kit/explainers.py
import pandas as pd
import numpy as np
import networkx as nx
import shap
from lingam import VARLiNGAM  # Make sure lingam is installed
from .modeling import train_model
import pandas as pd
import numpy as np
import networkx as nx
import shap


# causal_explainer_kit/explainers.py
from lingam import VARLiNGAM  # Make sure lingam is installed
import lime # New import
import lime.lime_tabular # New import
import matplotlib.pyplot as plt # New import
import os # New import

class CausalExplainer:
    def __init__(self, model_type="catboost", top_n_features=3, explainer_type="shap",
                 task_type="classification"):
        """
        Initialize the CausalExplainer.

        Parameters:
        - model_type (str): Type of predictive model ('randomforest', 'xgboost', etc.).
        - top_n_features (int): Number of top features to include in the explanation graph.
        - explainer_type (str): Feature attribution method ('shap' or 'lime').
        - task_type (str): Type of task ('classification' or 'regression'). Default is 'classification'.
        """
        self.model_type = model_type.lower()
        self.top_n_features = top_n_features
        self.explainer_type = explainer_type.lower()
        self.task_type = task_type.lower()

        self.predictive_model = None
        self.causal_model = None
        self.full_causal_graph = None
        self.feature_names = None # Includes all columns from input data (features + Target)
        self.X_train = None # To store training features (X) for LIME
        self.y_train = None # To store training target (y) for LIME class names
        self.class_names = None # For LIME classification

        if self.explainer_type not in ["shap", "lime"]:
            raise ValueError("explainer_type must be 'shap' or 'lime'")
        if self.task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be 'classification' or 'regression'")

    def fit(self, train_data):
        """
        Fit the explainer on training data by training a predictive model and a causal model.

        Parameters:
        - train_data (pd.DataFrame): Training data with features and 'Target' column.
        """
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("train_data must be a pandas DataFrame")
        if "Target" not in train_data.columns:
            raise ValueError("train_data must contain a 'Target' column")

        X_train = train_data.drop(columns=["Target"])
        y_train = train_data["Target"]

        self.X_train = X_train # Store for LIME and consistent feature set
        self.y_train = y_train # Store for LIME

        if self.task_type == "classification":
            self.class_names = [str(c) for c in sorted(list(np.unique(y_train)))]

        model_result = train_model(self.model_type, X_train, y_train)
        if model_result is None:
            raise RuntimeError(f"Failed to train {self.model_type} model")
        self.predictive_model = model_result["model"]

        self.causal_model = VARLiNGAM()
        try:
            self.causal_model.fit(train_data.values)
        except np.linalg.LinAlgError:
            # Add small noise to handle singularity
            train_data_noisy = train_data + np.random.normal(0, 1e-4, train_data.shape)
            self.causal_model.fit(train_data_noisy.values)

        self.feature_names = train_data.columns.tolist()
        adjacency_matrices = self.causal_model.adjacency_matrices_
        self.full_causal_graph = nx.DiGraph()
        self.full_causal_graph.add_nodes_from(self.feature_names)

        # LiNGAM adjacency_matrix_[i, j] means feature_j -> feature_i
        for adj_matrix in adjacency_matrices: # Iterate over lag matrices if multiple
            for i in range(len(self.feature_names)): # Target node index
                for j in range(len(self.feature_names)): # Source node index
                    weight = adj_matrix[i, j]
                    if weight != 0:
                        # Add edge from feature_j to feature_i
                        source_node = self.feature_names[j]
                        target_node = self.feature_names[i]
                        # If edge exists from a different lag, this might update/overwrite weight.
                        # Here, add_edge will update attributes if the edge already exists.
                        self.full_causal_graph.add_edge(source_node, target_node, weight=weight)

    def predict(self, test_data):
        """
        Generate a causal graph for a specific test instance.

        Parameters:
        - test_data (pd.DataFrame or pd.Series): Single test instance.

        Returns:
        - nx.DiGraph: Subgraph showing causal relationships among top features and Target.
        """
        if self.predictive_model is None or self.full_causal_graph is None or self.X_train is None:
            raise RuntimeError("Explainer not fitted. Call fit() first.")

        if isinstance(test_data, pd.Series):
            test_data_df = test_data.to_frame().T
        elif isinstance(test_data, pd.DataFrame):
            test_data_df = test_data.copy() # Use a copy to avoid modifying original
        else:
            raise ValueError("test_data must be a pandas DataFrame or Series")
        
        if test_data_df.empty:
            raise ValueError("test_data is empty.")

        expected_features = self.X_train.columns.tolist()
        
        # Align test_data_df columns with X_train columns
        test_data_for_explainer = test_data_df.reindex(columns=expected_features, fill_value=0)

        if self.explainer_type == "shap":
            try:
                explainer = shap.TreeExplainer(self.predictive_model)
            except Exception: 
                explainer = shap.KernelExplainer(self.predictive_model.predict_proba if self.task_type == "classification" else self.predictive_model.predict, 
                                                 shap.sample(self.X_train, 50)) 

            shap_values = explainer.shap_values(test_data_for_explainer)

            if self.task_type == "classification" and isinstance(shap_values, list) and len(shap_values) > 1:
                feature_importance = np.abs(shap_values[1]).mean(axis=0)
            elif isinstance(shap_values, list) and len(shap_values) == 1: 
                 feature_importance = np.abs(shap_values[0]).mean(axis=0)
            else: 
                feature_importance = np.abs(shap_values).mean(axis=0)
            
            feature_importance_df = pd.DataFrame(
                {"feature": expected_features, "importance": feature_importance}
            )
        elif self.explainer_type == "lime":
            if self.task_type == "classification":
                if not hasattr(self.predictive_model, 'predict_proba'):
                    raise AttributeError(f"Predictive model {self.model_type} lacks 'predict_proba' for LIME classification.")
                predict_fn = lambda x: self.predictive_model.predict_proba(pd.DataFrame(x, columns=expected_features))
                lime_mode = 'classification'
            else: 
                if not hasattr(self.predictive_model, 'predict'):
                    raise AttributeError(f"Predictive model {self.model_type} lacks 'predict' for LIME regression.")
                predict_fn = lambda x: self.predictive_model.predict(pd.DataFrame(x, columns=expected_features))
                lime_mode = 'regression'

            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.X_train.values,
                feature_names=expected_features,
                class_names=self.class_names,
                mode=lime_mode,
                discretize_continuous=True
            )

            test_instance_numpy = test_data_for_explainer.iloc[0].values

            if self.task_type == "classification":
                num_classes = len(self.predictive_model.classes_) if hasattr(self.predictive_model, 'classes_') else (len(self.class_names) if self.class_names else 0)
                label_to_explain = (1,) if num_classes > 1 else (0,)
                
                explanation = lime_explainer.explain_instance(
                    data_row=test_instance_numpy,
                    predict_fn=predict_fn,
                    num_features=len(expected_features),
                    labels=label_to_explain
                )
                lime_scores = dict(explanation.as_list(label=label_to_explain[0]))
            else: 
                explanation = lime_explainer.explain_instance(
                    data_row=test_instance_numpy,
                    predict_fn=predict_fn,
                    num_features=len(expected_features)
                )
                lime_scores = dict(explanation.as_list())
            
            feature_importance_values = [lime_scores.get(f, 0.0) for f in expected_features]
            feature_importance_df = pd.DataFrame({
                "feature": expected_features,
                "importance": np.abs(feature_importance_values)
            })
        else:
            raise ValueError(f"Invalid explainer_type: {self.explainer_type}")

        top_features = (
            feature_importance_df.sort_values("importance", ascending=False)
            .head(self.top_n_features)["feature"]
            .tolist()
        )

        nodes_to_include = top_features + ["Target"]
        valid_nodes_to_include = [n for n in nodes_to_include if n in self.full_causal_graph]
        subgraph = self.full_causal_graph.subgraph(valid_nodes_to_include).copy()

        return subgraph









# ---------------------------------------------------------------------------------------------------------

# class CausalExplainer:
#     def __init__(self, model_type="catboost", top_n_features=3, explainer_type="shap"):
#         """
#         Initialize the CausalExplainer.

#         Parameters:
#         - model_type (str): Type of predictive model ('randomforest', 'xgboost', etc.).
#         - top_n_features (int): Number of top features to include in the explanation graph.
#         - explainer_type (str): Feature attribution method ('shap' or 'lime').
#         """
#         self.model_type = model_type.lower()
#         self.top_n_features = top_n_features
#         self.explainer_type = explainer_type.lower()
#         self.predictive_model = None
#         self.causal_model = None
#         self.full_causal_graph = None
#         self.feature_names = None

#     def fit(self, train_data):
#         """
#         Fit the explainer on training data by training a predictive model and a causal model.

#         Parameters:
#         - train_data (pd.DataFrame): Training data with features and 'Target' column.
#         """
#         # Validate input
#         if not isinstance(train_data, pd.DataFrame):
#             raise ValueError("train_data must be a pandas DataFrame")
#         if "Target" not in train_data.columns:
#             raise ValueError("train_data must contain a 'Target' column")

#         # Split features and target
#         X_train = train_data.drop(columns=["Target"])
#         y_train = train_data["Target"]

#         # Train predictive model using your train_model function
#         model_result = train_model(self.model_type, X_train, y_train)
#         if model_result is None:
#             raise RuntimeError(f"Failed to train {self.model_type} model")
#         self.predictive_model = model_result["model"]

#         # Fit causal model (VARLiNGAM) on all variables including Target
#         self.causal_model = VARLiNGAM()
#         try:
#             self.causal_model.fit(train_data.values)
#         except np.linalg.LinAlgError:
#             # Add small noise to handle singularity, as in create_causal_network
#             train_data_noisy = train_data + np.random.normal(0, 1e-4, train_data.shape)
#             self.causal_model.fit(train_data_noisy.values)

#         # Build full causal graph
#         self.feature_names = train_data.columns.tolist()
#         adjacency_matrices = self.causal_model.adjacency_matrices_
#         self.full_causal_graph = nx.DiGraph()
#         self.full_causal_graph.add_nodes_from(self.feature_names)

#         for lag_index, adj_matrix in enumerate(adjacency_matrices):
#             for i in range(len(self.feature_names)):
#                 for j in range(len(self.feature_names)):
#                     weight = adj_matrix[i, j]
#                     if weight != 0:
#                         self.full_causal_graph.add_edge(
#                             self.feature_names[i], self.feature_names[j], weight=weight
#                         )

#     def predict(self, test_data):
#         """
#         Generate a causal graph for a specific test instance.

#         Parameters:
#         - test_data (pd.DataFrame or pd.Series): Single test instance.

#         Returns:
#         - nx.DiGraph: Subgraph showing causal relationships among top features and Target.
#         """
#         # Convert test_data to DataFrame if Series
#         if isinstance(test_data, pd.Series):
#             test_data = test_data.to_frame().T
#         elif not isinstance(test_data, pd.DataFrame):
#             raise ValueError("test_data must be a pandas DataFrame or Series")

#         # Ensure test_data matches training features (excluding Target initially)
#         expected_features = [f for f in self.feature_names if f != "Target"]
#         if list(test_data.columns) != expected_features:
#             test_data = test_data.reindex(columns=expected_features, fill_value=0)

#         # Get feature importances using SHAP
#         if self.explainer_type == "shap":
#             explainer = shap.TreeExplainer(self.predictive_model)
#             shap_values = explainer.shap_values(test_data)
#             # For binary classification, use class 1 (positive class) SHAP values
#             if isinstance(shap_values, list):
#                 feature_importance = np.abs(shap_values[1]).mean(axis=0)
#             else:
#                 feature_importance = np.abs(shap_values).mean(axis=0)
#             feature_importance_df = pd.DataFrame(
#                 {"feature": expected_features, "importance": feature_importance}
#             )
#         elif self.explainer_type == "lime":
#             # Note: LIME requires training data for the explainer, which we don't store
#             # For simplicity, assume SHAP here; extend for LIME if needed
#             raise NotImplementedError("LIME explainer_type is not implemented yet")
#         else:
#             raise ValueError("explainer_type must be 'shap' or 'lime'")

#         # Select top N features
#         top_features = (
#             feature_importance_df.sort_values("importance", ascending=False)
#             .head(self.top_n_features)["feature"]
#             .tolist()
#         )

#         # Nodes to include in subgraph: top features + Target
#         nodes_to_include = top_features + ["Target"]

#         # Extract subgraph induced by these nodes
#         subgraph = self.full_causal_graph.subgraph(nodes_to_include).copy()

#         return subgraph
