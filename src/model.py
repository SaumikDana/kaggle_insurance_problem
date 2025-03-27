from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


def define_model():

    """Define robust model configurations optimized for high-missing-data scenarios."""
    
    # Base configurations
    base_configs = {
        # Conservative RandomForest for general use
        'rf_robust': RandomForestRegressor(
            n_estimators=100,
            max_depth=6,                  # Reduced depth
            min_samples_leaf=50,          # Increased to prevent overfitting
            min_samples_split=100,        # Added to ensure robust splits
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,              # Enable out-of-bag scoring
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
