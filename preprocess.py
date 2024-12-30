import numpy as np

def preprocess_insurance_data(df):
    df = df.copy()
    ids = df['id'].copy()
    
    # Drop unnecessary columns
    df.drop(['id', 'Policy Start Date'], axis=1, inplace=True)
    df.replace([float('inf'), float('-inf')], np.nan, inplace=True)
    
    print("\nNaN values before imputation:")
    print(df.isna().sum())
    
    # First handle categorical variables using complete columns
    # Group by complete columns that might have logical relationships
    
    # Occupation (358,075 missing)
    # Education Level and Location are complete and likely related to occupation
    df['Occupation'] = df.groupby(['Education Level', 'Location'], observed=True)['Occupation'].transform(
        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'Unknown')
    )
    
    # Health Score (74,076 missing) could relate to Exercise Frequency and Smoking Status
    df['Health Score'] = df.groupby(['Exercise Frequency', 'Smoking Status'], observed=True)['Health Score'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Credit Score (137,882 missing) might relate to Property Type and Policy Type
    df['Credit Score'] = df.groupby(['Property Type', 'Policy Type'], observed=True)['Credit Score'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Marital Status (18,529 missing) might relate to Property Type
    df['Marital Status'] = df.groupby('Property Type', observed=True)['Marital Status'].transform(
        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'Unknown')
    )
    
    # For remaining numeric columns, use grouping by Policy Type and Property Type
    numeric_cols = ['Age', 'Annual Income', 'Number of Dependents', 'Previous Claims', 
                   'Vehicle Age', 'Insurance Duration']
    
    for col in numeric_cols:
        df[col] = df.groupby(['Policy Type', 'Property Type'], observed=True)[col].transform(
            lambda x: x.fillna(x.median())
        )
    
    # Customer Feedback can use Policy Type
    df['Customer Feedback'] = df.groupby('Policy Type', observed=True)['Customer Feedback'].transform(
        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'Unknown')
    )
    
    print("\nNaN values after imputation:")
    print(df.isna().sum())
    
    return df, ids

