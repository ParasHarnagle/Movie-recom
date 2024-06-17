import mlflow
from src.data_processing import load_preprocess_data
from src.model_training import train_model
from src.model_deployment import deploy_model
from src.prediction import predict_recommendations

if __name__ == "__main__":
    #mlflow setup
    mlflow.set_experiment("movie_recommender")
    mlflow.sklearn.autolog()

    #load data
    movies_df = load_preprocess_data()

    #train model
    tfidf, tfidf_matrix = train_model(movies_df,{"max_features":5000})

    # deploy model
    model_uri = mlflow.get_artifact_uri("model")
    deploy_model(model_uri,"movie_recommnder_contect")

    #example preds
    user_genres = ["Action", "Sci-Fi"]
    user_actors = ["Tom Hanks", "Leonardo DiCaprio"]
    user_directors = ["Christopher Nolan"]
    recommendations = predict_recommendations(
        f"models:/movie_recommender_content/Production", "data/imdb_movies.csv", user_genres, user_actors, user_directors
    )

    print(f"Recommendations for user: ")
    for movie_id in recommendations:
        print(f"- {movies_df.loc[movie_id]['Name']}")