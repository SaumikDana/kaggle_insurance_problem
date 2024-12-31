import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize(df):

    # Separate numerical features excluding 'Premium Amount'
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_features = numerical_features.drop('Premium Amount')  # Ensure the target is not included

    # Visualizations
    # 1. Distribution of Numerical Features (Grid Layout)
    plt.figure(figsize=(16, 12))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(3, 3, i)
        df[feature].replace([float('inf'), float('-inf')], np.nan, inplace=True)
        sns.histplot(df[feature], kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Distribution of Categorical Features (Grid Layout)
    categorical_features = df.select_dtypes(include=['object']).columns
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(categorical_features, 1):
        plt.subplot(4, 3, i)
        sns.countplot(data=df, x=feature)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # 3. Distribution of Premium
    plt.figure(figsize=(10, 6))
    df['Premium Amount'].replace([float('inf'), float('-inf')], np.nan, inplace=True)
    sns.histplot(df['Premium Amount'], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of Premium Amount')
    plt.xlabel('Premium Amount')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()