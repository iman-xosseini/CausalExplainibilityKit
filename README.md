# Causal Explainability Kit


A Python toolkit for generating and visualizing causal explanations for machine learning models, 
focusing on identifying influential features and their causal relationships concerning a target variable.

## Features

-   Train various predictive models (RandomForest, XGBoost, LightGBM, CatBoost).
-   Employ VARLiNGAM for causal discovery.
-   Utilize SHAP for feature importance attribution.
-   Generate and visualize causal graphs focusing on top influential features and a target variable.
-   Customizable graph visualization.

## Installation

1.  **Clone the repository (if you host it on Git):**
    ```bash
    git clone [https://github.com/yourusername/CausalExplainabilityKit.git](https://github.com/yourusername/CausalExplainabilityKit.git)
    cd CausalExplainabilityKit
    ```

2.  **Install using pip:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
    Then install the package:
    ```bash
    pip install .
    ```
    This will also install all dependencies listed in `requirements.txt`.

    Alternatively, for development mode:
    ```bash
    pip install -e .
    ```

## Dependencies

The main dependencies are:
-   pandas
-   numpy
-   scikit-learn
-   xgboost
-   lightgbm
-   catboost
-   shap
-   lime
-   lingam
-   networkx
-   matplotlib

Refer to `requirements.txt` for a full list of dependencies and their versions.


## Usage

Here's a basic example of how to use the `CausalExplainabilityKit`:

```python
# 1. Import necessary libraries
import pandas as pd
import sys
sys.path.append('..')
from causal_explainer_kit.data_processing import load_uci_repository, generate_random_row
from causal_explainer_kit.explainers import CausalExplainer
from causal_explainer_kit.visualization import display_causal_graph_customized
from sklearn.model_selection import train_test_split


# 2. Load data
df = load_uci_repository(scaling=True)


# 3. Split the dataset into X and y
X = df.drop(columns=['Failure_Type'])
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# 4. Initialize and fit explainer
explainer = CausalExplainer(model_type='catboost', top_n_features=3, explainer_type='shap')


# 5. Train the explainer (uses the whole dataset if train=1, else X_train).
train = 1
explainer.fit(train_data=(X if train == 1 else X_train))


# 6. Generate random instance based on the whole dataset or on X_test
dataframe_for_instance = X if train == 1 else X_test
test_instance = generate_random_row(dataframe=dataframe_for_instance, target_name='Target', target_value=1)
test_features = test_instance.drop(columns=['Target'])


# 7. Predict causal graph
causal_graph = explainer.predict(test_features)


# 8. Visualize the graph
display_causal_graph_customized(causal_graph=causal_graph,
                                target_node_name="Target",  
                                seed=120,                              
                                node_base_size=1800,
                                plot_name="plot_442.png"
)
```