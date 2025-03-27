import setup_path
from src.segment_data import segment_test_data
import pandas as pd


def predict_and_export(
        test_df, 
        test_ids, 
        segment_results, 
        segment_function, 
        output_file='predicted_premiums.csv'
    ):

    """
    Predict and export results for test data using the trained models.

    Parameters:
        test_df (pd.DataFrame): The processed test data
        test_ids (pd.Series): Corresponding IDs for tracking
        segment_results (dict): Trained models by segment
        segment_function (callable): Function to segment the data
        categorical_features (list): List of categorical features
        output_file (str): File name to export predictions
    """
    predictions = []

    # Segment test data
    segments = segment_test_data(test_df, segment_function)

    for segment_name, segment_df in segments.items():

        if not segment_df.empty:

            try:
                model = segment_results.get(segment_name)
                if model is None:
                    print(f"No trained model for segment: {segment_name}")
                    continue

                segment_ids = test_ids.loc[segment_df.index]
                segment_preds = model['model'].predict(segment_df)

                predictions.extend(zip(segment_ids, segment_preds))

            except Exception as e:
                print(f"Error in segment {segment_name}: {e}")

    predictions_df = pd.DataFrame(predictions, columns=['id', 'Premium Amount'])

    # Average over duplicate IDs
    predictions_df = predictions_df.groupby('id', as_index=False)['Premium Amount'].mean()

    predictions_df.to_csv(output_file, index=False)
    print(f"Predictions exported to {output_file}.")

    return