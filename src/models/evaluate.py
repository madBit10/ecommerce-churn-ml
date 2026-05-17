from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate(model, X_test, y_test) -> dict:
    """
    Evaluates the trained model on test data.

    Args: 
        model: Trained Random Forest model
        X_test: Test features
        y_test: True test labels

    Returns:
        metrics: Dictionary of evaluation results 
    """

    logger.info("Evaluating model...")

    y_pred = model.predict(X_test)

    # probablities
    y_proba_pos = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test,y_pred).tolist()
    roc_auc = roc_auc_score(y_test, y_proba_pos)
    pr_auc = average_precision_score(y_test, y_proba_pos)

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Recall: {report['1']['recall']:.4f}")
    logger.info(f"F1: {report['1']['f1-score']:.4f}")
    logger.info(f"roc_auc: {roc_auc:.4f}")
    logger.info(f"pr_auc: {pr_auc:.4f}")

    return metrics