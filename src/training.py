from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
import numpy as np


def create_segment_pipeline(config, categorical_features, category_mappings, numeric_features):

    categorical_transformer = OneHotEncoder(
        categories=[category_mappings[col] for col in categorical_features],
        sparse_output=False,
        handle_unknown='ignore'
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        sparse_threshold=0
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', config['model'])
    ])

    return pipeline


def evaluate_predictions(y_true, y_pred):

    return {
        'r2_score': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'median_ae': np.median(np.abs(y_true - y_pred))
    }


def train_segment_model(X_seg, y_seg, config, categorical_features, category_mappings):

    numeric_features = X_seg.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if 'Premium Amount' in numeric_features:
        numeric_features.remove('Premium Amount')

    pipeline = create_segment_pipeline(config, categorical_features, category_mappings, numeric_features)

    X_train, X_test, y_train, y_test = train_test_split(X_seg, y_seg, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    train_metrics = evaluate_predictions(y_train, train_pred)
    test_metrics = evaluate_predictions(y_test, test_pred)

    cv_scores = cross_val_score(pipeline, X_seg, y_seg, cv=5, scoring=make_scorer(r2_score), n_jobs=-1)

    return {
        'model': pipeline,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_data': (X_test, y_test)
    }

