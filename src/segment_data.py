import numpy as np


def define_segments(df):

    """Define customer segments with fuzzy boundaries and consideration of data quality."""
    
    # Calculate reliability scores based on imputed values
    reliability_score = 1.0

    for col in ['Occupation_is_missing', 'Previous Claims_is_missing', 'Credit Score_is_missing']:

        if col in df.columns:

            reliability_score -= df[col] * 0.1
    
    segments = {
        'High_Value_Property': (
            (df['Property Type'].isin(['House', 'Condo'])) &
            (df['Annual Income'] >= df['Annual Income'].quantile(0.55)) &
            ((df['Policy Type'] == 'Premium') | 
             (df['Policy Type'] == 'Standard')) &
            (df['Credit Score'] > df['Credit Score'].quantile(0.45)) &
            (reliability_score >= 0.8)  # Only include highly reliable records
        ),
        
        'Low_Risk_Premium': (
            (df['Credit Score'] > df['Credit Score'].quantile(0.5)) &
            (df['Health Score'] > df['Health Score'].quantile(0.5)) &
            (df['Insurance Duration'] > 1) &
            (df['Annual Income'] > df['Annual Income'].quantile(0.4)) &
            (reliability_score >= 0.7)
        ),
        
        'Healthy_Professional': (
            ((df['Age'] <= df['Age'].quantile(0.4)) |
             (df['Exercise Frequency'].isin(['Daily', 'Weekly']))) &
            (df['Annual Income'] > df['Annual Income'].quantile(0.45)) &
            (df['Health Score'] > df['Health Score'].quantile(0.5)) &
            (reliability_score >= 0.7)
        ),
        
        'Family_Premium': (
            (df['Number of Dependents'] >= 1) &
            (df['Location'].isin(['Suburban', 'Rural'])) &
            (df['Marital Status'] == 'Married') &
            (df['Annual Income'] > df['Annual Income'].quantile(0.4)) &
            (reliability_score >= 0.8)
        ),
        
        'Basic_Coverage': (
            ((df['Annual Income'] <= df['Annual Income'].quantile(0.35)) |
             (df['Previous Claims'] >= 2) |
             (df['Credit Score'] < df['Credit Score'].quantile(0.35))) &
            (df['Policy Type'] == 'Basic')
        )
    }
    
    return segments


def segment_data(df):

    segments = define_segments(df)

    # Create a mask for all data assigned to a segment
    assigned_mask = np.zeros(len(df), dtype=bool)

    for mask in segments.values():

        assigned_mask |= mask

    # Create a default segment for unassigned data
    segments['Default_Segment'] = ~assigned_mask

    # Print distribution for debugging
    total_records = len(df)
    print("\nSegment Distribution:")
    total_assigned = 0

    for name, mask in segments.items():

        segment_size = mask.sum()
        total_assigned += segment_size
        percentage = (segment_size / total_records) * 100
        print(f"{name}: {segment_size:,} ({percentage:.1f}%)")

    # Additional debug info
    print(f"\nTotal records: {total_records:,}")
    print(f"Total assigned: {total_assigned:,}")
    print(f"Records per segment on average: {total_assigned/len(segments):,.1f}")

    return segments
