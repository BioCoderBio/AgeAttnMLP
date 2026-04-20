from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def get_svm_model():
    """
    Initializes and returns an SVM model with a StandardScaler in a pipeline.
    
    Returns:
        Pipeline: A scikit-learn Pipeline containing a StandardScaler and an SVC model.
    """
    # SVMs are sensitive to feature scaling, so a StandardScaler is used.
    # The pipeline ensures that the scaler is fit on the training data and
    # used to transform both the training and validation data.
    
    # class_weight='balanced': Handles class imbalance.
    # probability=True: Enables probability estimates for ROC-AUC calculation.
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            kernel='rbf', 
            C=1.0, 
            gamma='scale', 
            class_weight='balanced', 
            probability=True, 
            random_state=42
        ))
    ])
    return model
