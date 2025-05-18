# causal_explainer_kit/modeling.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier




def train_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: pd.Series,
    n_folds=10,
):
    try:
        model_type = model_type.lower()
        model_switch = {
            "randomforest": RandomForestClassifier(),
            "xgboost": XGBClassifier(
                eval_metric="logloss",  # Specify evaluation metric to avoid warnings
            ),
            "lgbmclassifier": LGBMClassifier(
                boosting_type="gbdt",  # Gradient Boosting Decision Trees
                n_estimators=100,  # Number of boosting rounds
                learning_rate=0.1,  # Shrinkage rate
                max_depth=-1,  # Unlimited depth
            ),
            "catboost": CatBoostClassifier(
                iterations=100,  # Number of boosting iterations
                learning_rate=0.1,
                depth=6,
                verbose=False,  # Suppress verbose output
            ),
        }

        # Get the model based on model_type
        model = model_switch.get(model_type)

        if model is None:
            raise AttributeError("Invalid model type")

        # Set up k-fold cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="roc_auc")

        # Train the final model on the entire training set
        model.fit(X_train, y_train)

        # Return both the model and cross-validation results
        return {
            "model": model,
            "cv_scores": cv_scores,
            "mean_cv_score": np.mean(cv_scores),
            "std_cv_score": np.std(cv_scores),
        }

    except AttributeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None







# -------------------------------------------------------------------------------------------------------------------------------


# def train_model(
#     model_type: str,
#     X_train: np.ndarray,
#     y_train: pd.Series,
#     n_folds: int = 10,
# ) -> dict | None:
#     """
#     Trains a specified classification model using k-fold cross-validation.

#     Args:
#         model_type (str): Type of model ('randomforest', 'xgboost', 'lgbmclassifier', 'catboost').
#         X_train (np.ndarray): Training features.
#         y_train (pd.Series): Training target.
#         n_folds (int): Number of folds for cross-validation.

#     Returns:
#         dict | None: A dictionary containing the trained model, CV scores,
#                       mean CV score, and std CV score, or None if an error occurs.
#     """
#     try:
#         model_type_lower = model_type.lower()
#         model_switch = {
#             "randomforest": RandomForestClassifier(random_state=42),
#             "xgboost": XGBClassifier(
#                 eval_metric="logloss", use_label_encoder=False, random_state=42
#             ), # Added random_state and use_label_encoder
#             "lgbmclassifier": LGBMClassifier(
#                 boosting_type="gbdt",
#                 n_estimators=100,
#                 learning_rate=0.1,
#                 max_depth=-1,
#                 random_state=42,
#                 verbose=-1 # Suppress LightGBM verbosity
#             ),
#             "catboost": CatBoostClassifier(
#                 iterations=100,
#                 learning_rate=0.1,
#                 depth=6,
#                 verbose=False,
#                 random_state=42
#             ),
#         }

#         model = model_switch.get(model_type_lower)

#         if model is None:
#             # Try to provide a more informative error, e.g. list available models
#             available_models = ", ".join(model_switch.keys())
#             raise AttributeError(f"Invalid model type: '{model_type}'. Available models are: {available_models}")

#         kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
#         cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="roc_auc")
#         model.fit(X_train, y_train)

#         return {
#             "model": model,
#             "cv_scores": cv_scores,
#             "mean_cv_score": np.mean(cv_scores),
#             "std_cv_score": np.std(cv_scores),
#         }

#     except AttributeError as e:
#         print(f"Model Training Error: {e}")
#     except Exception as e:
#         print(f"Unexpected error during model training: {e}")
#     return None