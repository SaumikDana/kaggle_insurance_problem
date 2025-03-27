import numpy as np
import pandas as pd


def analyze_feature_importance(model):

    feature_names = []

    num_features = model.named_steps['preprocessor'].transformers_[0][2]
    feature_names.extend(num_features)

    cat_features = model.named_steps['preprocessor'].transformers_[1][2]

    if len(cat_features) > 0:

        encoder = model.named_steps['preprocessor'].named_transformers_['cat']

        if hasattr(encoder, 'get_feature_names_out'):
            encoded = encoder.get_feature_names_out(cat_features)
            feature_names.extend(encoded)

    if hasattr(model.named_steps['regressor'], 'feature_importances_'):

        importances = model.named_steps['regressor'].feature_importances_

    else:

        importances = np.mean([
            est.feature_importances_ 
            for name, est in model.named_steps['regressor'].estimators_
            if hasattr(est, 'feature_importances_')
        ], axis=0)

    return pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)


def get_segment_predictions(segment_results, segment_name, X_new):

    if segment_name not in segment_results:

        raise ValueError(f"No model found for segment: {segment_name}")

    return segment_results[segment_name]['model'].predict(X_new)


def get_feature_importance(segment_results, segment_name, categorical_features):

    if segment_name not in segment_results:

        raise ValueError(f"No model found for segment: {segment_name}")

    return analyze_feature_importance(segment_results[segment_name]['model'], categorical_features)


def get_segment_metrics(segment_results, segment_name):

    if segment_name not in segment_results:

        raise ValueError(f"No results found for segment: {segment_name}")

    res = segment_results[segment_name]

    return {
        'train_metrics': res['train_metrics'],
        'test_metrics': res['test_metrics'],
        'cv_scores': res['cv_scores'],
        'cv_mean': res['cv_mean'],
        'cv_std': res['cv_std']
    }

