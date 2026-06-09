from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_features(text_data):
    
    tfidf = TfidfVectorizer(
        max_features=5000
    )

    X = tfidf.fit_transform(text_data)

    return X, tfidf