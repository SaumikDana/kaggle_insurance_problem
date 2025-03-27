from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


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


def initialize_metadata(df):

    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if 'Premium Amount' in categorical_features:

        categorical_features.remove('Premium Amount')
    
    category_mappings = {col: sorted(df[col].unique()) for col in categorical_features}

    return categorical_features, category_mappings


def define_model():

    """Define robust model configurations optimized for high-missing-data scenarios."""
    
    base_configs = {

        # Conservative RandomForest for general use
        'rf_robust': RandomForestRegressor(
            n_estimators=100,
            max_depth=6,                  # Reduced depth
            min_samples_leaf=50,          # Increased to prevent overfitting
            min_samples_split=100,        # Added to ensure robust splits
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,               # Enable out-of-bag scoring
            n_jobs=-1,
            random_state=42
        ),
        
        # XGBoost with strong regularization
        'xgb_conservative': XGBRegressor(
            n_estimators=100,
            max_depth=4,                  # Very shallow trees
            learning_rate=0.01,           # Slower learning rate
            subsample=0.7,                # Reduced sample size
            colsample_bytree=0.7,         # Feature subsampling
            min_child_weight=10,          # Increased to prevent overfitting
            reg_alpha=1,                  # L1 regularization
            reg_lambda=2,                 # L2 regularization
            random_state=42
        ),
        
        # Gradient Boosting with focus on robustness
        'gbm_simple': GradientBoostingRegressor(
            n_estimators=80,
            max_depth=3,                  # Very shallow trees
            learning_rate=0.01,           # Slower learning rate
            subsample=0.7,                # Subsample for robustness
            min_samples_leaf=50,          # Conservative leaf size
            random_state=42
        )

    }

    # Segment-specific configurations
    model = {

        'High_Value_Property': {
            'model': base_configs['rf_robust'],
            'description': 'Robust RF for high-value properties'
        },

        'Low_Risk_Premium': {
            'model': base_configs['gbm_simple'],
            'description': 'Simple GBM for low-risk segment'
        },

        'Healthy_Professional': {
            'model': base_configs['rf_robust'],
            'description': 'Robust RF for professional segment'
        },

        'Family_Premium': {
            'model': base_configs['xgb_conservative'],
            'description': 'Conservative XGBoost for family segment'
        },

        'Senior_Premium': {
            'model': base_configs['gbm_simple'],
            'description': 'Simple GBM for senior segment'
        },

        'Basic_Coverage': {
            'model': base_configs['rf_robust'],
            'description': 'Robust RF for basic coverage'
        },

        'default': {
            'model': base_configs['rf_robust'],
            'description': 'Default robust RF model'
        }

    }

    return model


def train_all_segments(df, segments):

    segment_configs = define_model()

    categorical_features, category_mappings = initialize_metadata(df)

    feature_cols = [col for col in df.columns if col != 'Premium Amount']

    target_col = 'Premium Amount'

    segment_results = {}

    for name, mask in segments.items():

        print(f"\nProcessing {name} segment...")

        X_seg = df[feature_cols][mask]
        y_seg = df[target_col][mask]

        print(f"Segment Length {len(X_seg)}...")

        if len(X_seg) >= 100:

            config = segment_configs.get(name, segment_configs['default'])

            result = train_segment_model(X_seg, y_seg, config, categorical_features, category_mappings)

            segment_results[name] = result

            print(f"Train R2: {result['train_metrics']['r2_score']:.4f}")
            print(f"Test R2: {result['test_metrics']['r2_score']:.4f}")
            print(f"MAE: {result['test_metrics']['mae']:.2f}")
            print(f"Median AE: {result['test_metrics']['median_ae']:.2f}")
            print(f"MAPE: {result['test_metrics']['mape']:.2f}%")
            print(f"RMSE: {result['test_metrics']['rmse']:.2f}")
            print(f"CV R2: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})")

    performance_df = pd.DataFrame.from_dict({
        name: {
            'segment_size': len(df[segments[name]]),
            'train_r2': res['train_metrics']['r2_score'],
            'test_r2': res['test_metrics']['r2_score'],
            'cv_mean_r2': res['cv_mean'],
            'cv_std_r2': res['cv_std'],
            'mae': res['test_metrics']['mae'],
            'mape': res['test_metrics']['mape']
        }
        for name, res in segment_results.items()
    }, orient='index')

    print("\nSegment Performance Summary:")
    print(performance_df.sort_values('test_r2', ascending=False))

    return segment_results, performance_df

