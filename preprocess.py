import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------------------------------
# Multi-Column DistributionImputer
# -------------------------------------------------------
class DistributionImputer(BaseEstimator, TransformerMixin):
    """
    Imputer that randomly samples missing values from the empirical distribution
    of each column (can handle multiple columns of the same type).
    
    Parameters
    ----------
    col_type : {'continuous', 'categorical'}, default='continuous'
        'continuous': sample from raw numeric array of observed values per column
        'categorical': sample from frequency distribution per column
    random_state : int or None
        random seed for reproducibility
    """

    def __init__(self, col_type='continuous', random_state=None):
        self.col_type = col_type
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """
        X will be shape (n_samples, n_columns).
        We'll learn the distribution separately for each column.
        """
        X = self._as_numpy_array(X)
        self.n_cols_ = X.shape[1]
        self.rng_ = np.random.default_rng(self.random_state)

        # We'll store distribution info for each column in lists
        self._col_values = []
        self._col_categories = []
        self._col_probs = []

        for col_i in range(self.n_cols_):
            col_data = X[:, col_i]
            # Drop NaNs
            non_missing = col_data[~pd.isna(col_data)]

            if self.col_type == 'continuous':
                # Just store raw non-missing values for sampling later
                self._col_values.append(non_missing)
                self._col_categories.append(None)
                self._col_probs.append(None)
            else:
                # For categorical columns
                categories, counts = np.unique(non_missing, return_counts=True)
                probs = counts / counts.sum()

                self._col_values.append(None)
                self._col_categories.append(categories)
                self._col_probs.append(probs)

        return self

    def transform(self, X):
        """
        Impute missing values in each column by sampling from learned distribution.
        """
        X = self._as_numpy_array(X)
        for col_i in range(self.n_cols_):
            col_data = X[:, col_i]
            missing_mask = pd.isna(col_data)
            if not np.any(missing_mask):
                # no missing => skip
                continue

            n_missing = missing_mask.sum()
            if self.col_type == 'continuous':
                sample_pool = self._col_values[col_i]
                if sample_pool is not None and len(sample_pool) > 0:
                    X[missing_mask, col_i] = self.rng_.choice(sample_pool, size=n_missing, replace=True)
            else:
                cats = self._col_categories[col_i]
                probs = self._col_probs[col_i]
                if cats is not None and len(cats) > 0:
                    X[missing_mask, col_i] = self.rng_.choice(
                        cats, size=n_missing, replace=True, p=probs
                    )

        return X

    @staticmethod
    def _as_numpy_array(X):
        """Helper to ensure we have a 2D numpy array."""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        # If it's 1D, reshape to (n_samples, 1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X


# -------------------------------------------------------
# preprocess_data Function
# -------------------------------------------------------
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def preprocess_data(df):
    """
    Takes a DataFrame 'df' as input and returns:
      1) The 'id' column as a Series.
      2) The imputed DataFrame (with no missing values) + the original Premium Amount column.
         -- now with original dtypes re-applied where possible.
    """

    # -------------------------------------------------------
    # 0. Capture the original dtypes
    # -------------------------------------------------------
    original_dtypes = df.dtypes.to_dict()

    # -------------------------------------------------------
    # 1. Extract 'id' and 'Premium Amount' columns
    # -------------------------------------------------------
    id_col = df['id'].copy()
    premium_col = df['Premium Amount'].copy()

    # -------------------------------------------------------
    # 2. Drop unwanted columns
    # -------------------------------------------------------
    # 'errors="ignore"' just in case the columns don't exist
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
        'Previous Claims',      # 364,029 missing (0,1,2,...)
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
    # Continuous numeric pipeline
    continuous_numeric_transformer = Pipeline(steps=[
        ('dist_imputer', DistributionImputer(col_type='continuous', random_state=42))
    ])

    # Discrete numeric pipeline
    # You can treat them as continuous or categorical, depending on your preference
    discrete_numeric_transformer = Pipeline(steps=[
        ('dist_imputer', DistributionImputer(col_type='continuous', random_state=42))
    ])

    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('dist_imputer', DistributionImputer(col_type='categorical', random_state=42))
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
    # 5. Fit the transformer & transform
    # -------------------------------------------------------
    df_imputed_array = preprocessor.fit_transform(df)

    # Reconstruct a pandas DataFrame
    all_features = continuous_numeric_cols + discrete_numeric_cols + categorical_cols
    df_imputed = pd.DataFrame(df_imputed_array, columns=all_features)

    # -------------------------------------------------------
    # 6. Add 'Premium Amount' back
    # -------------------------------------------------------
    df_imputed['Premium Amount'] = premium_col.values

    # -------------------------------------------------------
    # 7. Re-apply the original dtypes where possible
    # -------------------------------------------------------
    for col, orig_dtype in original_dtypes.items():
        if col in df_imputed.columns:
            try:
                df_imputed[col] = df_imputed[col].astype(orig_dtype)
            except ValueError as e:
                # If casting fails (e.g., float -> int with decimals), you'll see a warning
                print(f"[WARNING] Could not cast '{col}' to {orig_dtype}: {e}")

    # Return both 'id' (as Series) and the final DataFrame (with Premium Amount)
    return df_imputed, id_col


# -------------------------------------------------------
# Example usage
# -------------------------------------------------------
if __name__ == "__main__":
    # Let's say you read your dataset:
    df_raw = pd.read_csv('train.csv')

    # Call the function
    df_clean, id_series = preprocess_data(df_raw)

    # Print checks
    print("Missing values after imputation:\n", df_clean.isnull().sum())
    print("\nSample of 'id' column:\n", id_series.head())
    print("\nImputed DataFrame (head):\n", df_clean.head())
