import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def preprocess_data(df: pd.DataFrame):
    """
    Takes a DataFrame 'df' as input and returns:
      1) The 'id' column as a Series.
      2) The imputed DataFrame (with no missing values) + the original Premium Amount column.
    """

    # ----------------------------
    # 1. Extract 'id' and 'Premium Amount' columns
    # ----------------------------
    id_col = df['id'].copy()
    premium_col = df['Premium Amount'].copy()

    # ----------------------------
    # 2. Drop unwanted columns
    # ----------------------------
    # 'errors="ignore"' just in case the columns don't exist in some scenario
    df.drop(columns=['id', 'Policy Start Date', 'Premium Amount'], inplace=True, errors='ignore')

    # -------------------------------------------------------
    # 3. Identify which columns are numeric vs. categorical
    # -------------------------------------------------------
    continuous_numeric_cols = [
        'Age',           # 18,705 missing
        'Annual Income', # 44,949 missing
        'Health Score',  # 74,076 missing
        'Credit Score',  # 137,882 missing
        'Vehicle Age'    # 6 missing
    ]
    discrete_numeric_cols = [
        'Number of Dependents', # 109,672 missing (0 to 4)
        'Previous Claims',      # 364,029 missing (mostly 0,1,2,...)
        'Insurance Duration'    # 1 missing (1 to 9 years)
    ]
    categorical_cols = [
        'Gender',           # 0 missing
        'Marital Status',   # 18,529 missing
        'Education Level',  # 0 missing
        'Occupation',       # 358,075 missing
        'Location',         # 0 missing
        'Policy Type',      # 0 missing
        'Customer Feedback',# 77,824 missing
        'Smoking Status',   # 0 missing
        'Exercise Frequency',# 0 missing
        'Property Type'     # 0 missing
    ]

    # -------------------------------------------------------
    # 4. Create Pipelines for numeric and categorical columns
    # -------------------------------------------------------
    continuous_numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    discrete_numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # Optionally, add an encoder (e.g., OneHotEncoder) if needed
        # ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cont_num', continuous_numeric_transformer, continuous_numeric_cols),
            ('disc_num', discrete_numeric_transformer, discrete_numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'  # keep other columns if present
    )

    # -------------------------------------------------------
    # 5. Fit the transformer on the data & transform
    # -------------------------------------------------------
    df_imputed_array = preprocessor.fit_transform(df)

    # Reconstruct a pandas DataFrame
    all_features = continuous_numeric_cols + discrete_numeric_cols + categorical_cols
    df_imputed = pd.DataFrame(df_imputed_array, columns=all_features)

    # -------------------------------------------------------
    # 6. Add 'Premium Amount' back to the imputed DataFrame
    # -------------------------------------------------------
    # Ensure the index alignment is correct; using .values ensures 
    # we just copy the series values in the same order.
    df_imputed['Premium Amount'] = premium_col.values

    # Return both 'id' (as Series) and the final DataFrame (with Premium Amount)
    return df_imputed, id_col

# ------------------------------
# Example of how you'd use this:
# ------------------------------
if __name__ == "__main__":
    # Let's say you read your dataset:
    df_raw = pd.read_csv('train.csv')

    # Call the function
    df_clean, id_series = preprocess_data(df_raw)

    # Print checks
    print("Missing values after imputation:\n", df_clean.isnull().sum())
    print("\nSample of 'id' column:\n", id_series.head())
    print("\nImputed DataFrame (head):\n", df_clean.head())
