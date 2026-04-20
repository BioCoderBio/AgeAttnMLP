from sklearn.ensemble import RandomForestClassifier

def get_rf_model():
    """
    Initializes and returns a RandomForestClassifier model with specific parameters.
    
    Returns:
        RandomForestClassifier: The scikit-learn RandomForestClassifier model.
    """
    # n_estimators=500: Use 500 trees to build the forest for robust feature sampling.
    # class_weight='balanced': Automatically handle class imbalance.
    # n_jobs=-1: Use all available CPU cores for acceleration.
    # random_state=42: Ensure reproducibility.
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,           # Allow trees to grow freely, relying on the ensemble to prevent overfitting.
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    return model
