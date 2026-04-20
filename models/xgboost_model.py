from xgboost import XGBClassifier

def get_xgboost_model(pos_weight=1.0):
    """
    Initializes and returns an XGBoost classifier model with specific parameters.
    
    Args:
        pos_weight (float): The weight for the positive class, used to handle class imbalance.

    Returns:
        XGBClassifier: The XGBoost classifier model.
    """
    # n_estimators=500: Maximum number of trees.
    # learning_rate=0.05: Step size shrinkage.
    # max_depth=6: Maximum depth of a tree.
    # subsample=0.8: Subsample ratio of the training instance.
    # colsample_bytree=0.8: Subsample ratio of columns when constructing each tree.
    # scale_pos_weight: Controls the balance of positive and negative weights.
    # early_stopping_rounds=50: Activates early stopping.
    # eval_metric='logloss': Evaluation metric for validation data.
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    return model
