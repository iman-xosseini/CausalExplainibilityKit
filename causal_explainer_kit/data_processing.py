# causal_explainer_kit/data_processing.py
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


def feature_engineering_uci_repository(
    df: pd.DataFrame, scaling: bool = True
) -> pd.DataFrame:
    """
    Placeholder for your feature engineering logic for the UCI repository data.
    You need to implement the actual feature engineering steps here.

    Args:
        df_pm (pd.DataFrame): Input DataFrame.
        scaling (bool): Whether to apply scaling.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """

    # scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    # label_encoder = LabelEncoder()
    encoder = OrdinalEncoder(categories=[["L", "M", "H"]])

    df.rename(
        columns={
            "Air temperature [K]": "Air_temperature",
            "Process temperature [K]": "Process_temperature",
            "Rotational speed [rpm]": "Rotational_speed",
            "Torque [Nm]": "Torque",
            "Tool wear [min]": "Tool_wear",
            "Failure Type": "Failure_Type",
        },
        inplace=True,
    )

    # Engineered Features
    df["Power_output"] = df["Torque"] * df["Rotational_speed"]
    df["wear_torque_product"] = df["Tool_wear"] * df["Torque"]
    df["Temp_difference"] = np.abs(df["Air_temperature"] - df["Process_temperature"])

    # exclude_columns = ["Type", "Failure Type"]
    mapping = {"L": 1, "M": 2, "H": 3}
    df["Type_n"] = df["Type"].map(mapping)
    df["Type_n"] = encoder.fit_transform(df[["Type"]]).astype(int)

    exclude_columns = ["Failure_Type", "Target", "Type_n"]

    df.drop(columns=["Product ID", "UDI", "Type"], inplace=True)

    # Columns to scale
    columns_to_scale = [col for col in df.columns if col not in exclude_columns]

    df_minmax = df.copy()
    df_minmax[columns_to_scale] = minmax_scaler.fit_transform(
        df_minmax[columns_to_scale]
    )
    df_minmax = pd.concat(
        [df_minmax[columns_to_scale], df_minmax[exclude_columns]], axis=1
    )

    if scaling:
        return df_minmax
    else:
        return df


def load_uci_repository(scaling: bool = True) -> pd.DataFrame:
    """
    Loads data from the UCI repository, performs initial filtering,
    and applies feature engineering.

    Args:
        filename (str): Path to the CSV file.
        scaling (bool): Whether to apply scaling during feature engineering.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    filename = "../data/predictive_maintenance.csv"
    try:
        df_pm = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        raise
    except Exception as e:
        print(f"Error reading CSV file {filename}: {e}")
        raise

    # Count rows before filtering
    rows_before = len(df_pm)

    # Remove rows where Target is 1 and Failure_Type is "No Failure"
    # Ensure columns exist before filtering
    if "Target" in df_pm.columns and "Failure Type" in df_pm.columns:
        df_pm = df_pm[
            ~((df_pm["Target"] == 1) & (df_pm["Failure Type"] == "No Failure"))
        ]
        rows_after = len(df_pm)
        print(
            f"Removed {rows_before - rows_after} rows where Target is 1 and Failure_Type is 'No Failure'"
        )
    else:
        print(
            "Warning: 'Target' or 'Failure Type' column not found. Skipping filtering step."
        )

    df = feature_engineering_uci_repository(df_pm, scaling)
    
    return df









# def load_uci_repository(filename: str, scaling: bool = True) -> pd.DataFrame:
#     """
#     Loads data from the UCI repository, performs initial filtering,
#     and applies feature engineering.

#     Args:
#         filename (str): Path to the CSV file.
#         scaling (bool): Whether to apply scaling during feature engineering.

#     Returns:
#         pd.DataFrame: Processed DataFrame.
#     """
#     try:
#         df_pm = pd.read_csv(filename)
#     except FileNotFoundError:
#         print(f"Error: The file {filename} was not found.")
#         raise
#     except Exception as e:
#         print(f"Error reading CSV file {filename}: {e}")
#         raise

#     # Count rows before filtering
#     rows_before = len(df_pm)

#     # Remove rows where Target is 1 and Failure_Type is "No Failure"
#     # Ensure columns exist before filtering
#     if "Target" in df_pm.columns and "Failure Type" in df_pm.columns:
#         df_pm = df_pm[
#             ~((df_pm["Target"] == 1) & (df_pm["Failure Type"] == "No Failure"))
#         ]
#         rows_after = len(df_pm)
#         print(
#             f"Removed {rows_before - rows_after} rows where Target is 1 and Failure_Type is 'No Failure'"
#         )
#     else:
#         print(
#             "Warning: 'Target' or 'Failure Type' column not found. Skipping filtering step."
#         )

#     df = feature_engineering_uci_repository(df_pm, scaling)
    
#     return df


def generate_random_row(
    dataframe: pd.DataFrame, target_name: str, target_value: int
) -> pd.DataFrame:
    """
    Selects a random row from the dataframe based on a specific target value.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        target_name (str): The name of the target column.
        target_value (int): The value in the target column to filter by.

    Returns:
        pd.DataFrame: A DataFrame containing a single randomly selected row.

    Raises:
        ValueError: If no rows are found with the specified target value
                    or if the target_name column does not exist.
    """
    if target_name not in dataframe.columns:
        raise ValueError(f"Target column '{target_name}' not found in DataFrame.")

    filtered_samples = dataframe[dataframe[target_name] == target_value]

    if len(filtered_samples) == 0:
        raise ValueError(f"No rows found where '{target_name}' is {target_value}.")

    random_sample = filtered_samples.sample(n=1)
    return random_sample
