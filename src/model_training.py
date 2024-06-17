import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer

def train_model(model_df,model_params):
    with mlflow.start_run():
        tfidf = TfidfVectorizer(stop_words="english", **model_params)
        tfidf_matrix = tfidf.fit_transform(model_df)

        mlflow.log_param("model_type","Content_Based")
        mlflow.sklearn.log_model(tfidf,"model")
        return tfidf, tfidf_matrix

