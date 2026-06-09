import pickle

from nlp_preprocessing import preprocess_text


# Load TF-IDF Vectorizer
with open("../models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)


# Load Best Model (Random Forest)
with open("../models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)


def predict_message(message):
    """
    Predict whether a message is Spam or Ham.

    Parameters:
        message (str): Input SMS message

    Returns:
        dict: Prediction and confidence score
    """

    # Preprocess text
    processed_text = preprocess_text(message)

    # Convert text to TF-IDF features
    vectorized_text = tfidf.transform([processed_text])

    # Prediction
    prediction = model.predict(vectorized_text)[0]

    # Prediction probabilities
    probability = model.predict_proba(vectorized_text)[0]

    # Highest probability as confidence
    confidence = float(round(max(probability), 4))

    if prediction == 1:
        return {
            "prediction": "Spam",
            "confidence": confidence
        }

    return {
        "prediction": "Ham",
        "confidence": confidence
    }


def inspect_message(message):
    """
    Debug helper to see how preprocessing transforms a message.
    """

    processed_text = preprocess_text(message)

    print("Original Message:")
    print(message)

    print("\nProcessed Message:")
    print(processed_text)

    return processed_text