import mlflow
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def predict_recommendations(model_uri,movies_data_path,user_geners,user_actors,
                            user_directors,num_recommendations=5):
    loaded_model = mlflow.sklearn.load_model(model_uri)
    tfidf = loaded_model

    movies_df = pd.read_csv(movies_data_path)
    user_profile = ' '.join(user_geners) + ' ' + ' '.join(user_actors) + ' ' + ' '.join(user_directors)

    similarities = movies_df.apply(
        lambda row: cal_similarity(row['content'],user_profile,tfidf), axis=1
    )
    sorted_similarities = similarities.sort_values(ascending=False)
    recommended_movies = sorted_similarities.index[:num_recommendations]
    return recommended_movies


    def cal_similarity(movie_content, user_profile,tfidf):
        similarity = cosine_similarity(
            tfidf.transform([movie_content]),
            tfidf.transform([user_profile])
        )[0,0]

        return similarity