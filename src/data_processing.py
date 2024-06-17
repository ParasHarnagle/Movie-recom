import pandas as pd
import json

def load_preprocess_data(data_path="data/movies.csv"):
    """
    Load and preprocess movie data from the csv file
    :param data_path:
    :return:
    """
    movies_df = pd.read_csv(data_path)
    movies_df['Genre'] = movies_df['Genre'].fillna('').apply(lambda x: x.split(', '))
    movies_df['Cast'] = movies_df['Cast'].fillna('').apply(lambda x: x.split(', '))
    movies_df['Director'] = movies_df['Director'].fillna('')
    movies_df['content'] = (movies_df['Genre'].apply(lambda x:' '.join(x)) + ' ' + movies_df['Director'] + ' ' +
                            movies_df['Cast'].apply(lambda x:' '.join(x)))

    return movies_df