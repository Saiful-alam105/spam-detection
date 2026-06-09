from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    return metrics


def print_classification_report(model,X_test,y_test):

    y_pred = model.predict(X_test)

    print(classification_report(y_test,y_pred))


def get_confusion_matrix(model,X_test,y_test):

    y_pred = model.predict(X_test)

    return confusion_matrix(y_test,y_pred)