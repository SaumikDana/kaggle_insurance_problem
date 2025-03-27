import numpy as np

# -------------------------------------------------------
# Impute missing values in numeric columns by sampling 
# from existing non-null values in each column.
# -------------------------------------------------------
def impute_numeric_random(df, columns, random_state=None):

    rng = np.random.default_rng(random_state)

    for col in columns:
        if col not in df.columns:
            continue
        missing_mask = df[col].isna()

        if missing_mask.any():
            # Get all non-missing values in the column
            values = df[col].dropna().values

            # Only proceed if there are values to sample from
            if len(values) > 0:
                # Randomly sample from the values (equal chance for each)
                # and fill the missing positions in the column
                df.loc[missing_mask, col] = rng.choice(
                    values,
                    size=missing_mask.sum(),
                    replace=True
                )

    return df

# -------------------------------------------------------
# Impute missing values in categorical columns by sampling 
# according to the frequency distribution of existing values.
# -------------------------------------------------------
def impute_categorical_random(df, columns, random_state=None):

    rng = np.random.default_rng(random_state)

    for col in columns:
        if col not in df.columns:
            continue
        missing_mask = df[col].isna()

        if missing_mask.any():
            # Get unique values and their counts from non-missing entries
            values, counts = np.unique(df[col].dropna(), return_counts=True)

            # Convert counts to probabilities (normalize)
            probs = counts / counts.sum()

            # Only proceed if there are values to sample from
            if len(values) > 0:
                # Sample missing values based on observed category frequencies
                df.loc[missing_mask, col] = rng.choice(
                    values,
                    size=missing_mask.sum(),
                    replace=True,
                    p=probs  # 👈 use frequency-based probabilities here
                )

    return df

# -------------------------------------------------------
# Preprocess input DataFrame:
# - Drop unused columns
# - Impute missing values (numeric & categorical)
# - Restore original dtypes
# - Reattach important columns ('id', 'Premium Amount')
# -------------------------------------------------------
def preprocess_data(df):
    # Backup original data types for later re-application
    original_dtypes = df.dtypes.to_dict()

    # Extract 'id' column
    id_col = df['id'].copy()

    # Extract target column if present
    flag = 'Premium Amount' in df.columns
    if flag:
        premium_col = df['Premium Amount'].copy()

    # Drop unused columns
    df = df.drop(columns=['id', 'Policy Start Date', 'Premium Amount'] if flag else ['id', 'Policy Start Date'], errors='ignore')

    # Define numeric and categorical feature sets
    numeric_cols = [
        'Age', 
        'Annual Income', 
        'Health Score', 
        'Credit Score', 
        'Vehicle Age',
        'Number of Dependents', 
        'Previous Claims', 
        'Insurance Duration'
    ]

    categorical_cols = [
        'Gender', 
        'Marital Status', 
        'Education Level', 
        'Occupation', 
        'Location',
        'Policy Type', 
        'Customer Feedback', 
        'Smoking Status', 
        'Exercise Frequency', 
        'Property Type'
    ]

    # Impute missing values using random sampling
    df = impute_numeric_random(df, numeric_cols, random_state=42)
    df = impute_categorical_random(df, categorical_cols, random_state=42)

    # Add back 'Premium Amount' if it was removed
    if flag:
        df['Premium Amount'] = premium_col.values

    # Restore original data types wherever possible
    for col, dtype in original_dtypes.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                print(f"[WARNING] Could not cast '{col}' to {dtype}: {e}")

    # Return the cleaned DataFrame and the id column separately
    return df, id_col
