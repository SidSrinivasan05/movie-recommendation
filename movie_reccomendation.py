import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

import numpy as np



import re

def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

df1 = pd.read_csv("movies.csv")

df1['genres'] = df1['genres'].str.split('|')

# Bersihkan judul film
df1['title'] = df1['title'].apply(clean_title)

# Perbarui movies_data
movies_data = df1[['movieId', 'title', 'genres']]

print(movies_data)


# # Load datasets
# movies_df = pd.read_csv('movies.csv')
# tags_df = pd.read_csv('tags.csv')

# # Clean and process the data
# movies_df['genre list'] = movies_df['genres'].apply(lambda x: x.split('|'))
# # grouped_df = tags_df.groupby('movieId', as_index=False).agg({'tag': list})
# # movies_df = movies_df.merge(grouped_df, on='movieId', how='left')
# # movies_df['tag'] = movies_df['tag'].apply(lambda x: x if isinstance(x, list) else [])
# movies_df['word'] = movies_df['genre list']
# movies_df['word'] = movies_df['word'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else '')

# movies_df = movies_df.drop(columns=['genres', 'genre list'], axis=1)
# print(movies_df)
# # # TF-IDF Vectorization
# vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = vectorizer.fit_transform(movies_df['word'])

# # # Calculate Cosine Similarity (using sparse matrix)
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix, dense_output=False)

# # # Convert sparse matrix to dense (if needed for easier manipulation)
# cosine_sim_dense = cosine_sim.toarray()

# # # Set the diagonal elements to zero
# np.fill_diagonal(cosine_sim_dense, 0)


# print(cosine_sim_dense)
# # Function to get movie recommendations based on a given movieId
# def get_movie_recommendations(movie_id, cosine_sim, top_n=5):
#     sim_scores = cosine_sim[movie_id].toarray().flatten()  # Convert sparse matrix to dense and flatten
#     sim_scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    
#     # Exclude the movie itself (index 0 is the movie itself)
#     sim_scores = [score for score in sim_scores if score[0] != movie_id]
    
#     # Get top N similar movies
#     top_similar_movies = sim_scores[:top_n]
    
#     # Get movie titles for the top similar movieIds
#     similar_movie_titles = [movies_df.iloc[i[0]]['title'] for i in top_similar_movies]
    
#     return similar_movie_titles

# # Get recommendations for a specific movie (e.g., movieId=1)
# recommended_movies = get_movie_recommendations(movie_id=1, cosine_sim=cosine_sim_dense, top_n=5)
# print("Recommended Movies:", recommended_movies)
