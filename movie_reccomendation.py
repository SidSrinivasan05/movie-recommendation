import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import re

og_movies = pd.read_csv("movies.csv")

og_movies['genres'] = og_movies['genres'].str.split('|')

# Cleaning titles
og_movies['title'] = og_movies['title'].apply(lambda title : re.sub("[^a-zA-Z0-9 ]", "", title))

movies_data = og_movies[['movieId', 'title', 'genres']]

full_list = []
for item in movies_data['genres']:
    item_list = [genre for genre in item]
    for i in item_list:
        full_list.append(i)
    
unique_genres = pd.Series(full_list).drop_duplicates().reset_index(drop=True)


genre_movies = movies_data[~movies_data['genres'].apply(lambda x: '(no genres listed)' in x)]

genre_movies.to_csv("movies_catalog.csv", index=False)

def search_title(title, top_k=5, movie_data=genre_movies):
    # Clean the title input
    cleaned_title = re.sub("[^a-zA-Z0-9 ]", "", title).lower()
    
    vectorizer_title = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_title = vectorizer_title.fit_transform(movie_data['title'])
    
    # Transform the query title into its tf-idf vector representation
    query_vec = vectorizer_title.transform([cleaned_title])
    
    # Compute cosine similarity between the input title and all movie titles
    similarity = cosine_similarity(query_vec, tfidf_title).flatten()
    
    # Get the indices of the top k most similar movies
    indices = np.argpartition(similarity, -top_k)[-top_k:]
    
    # Collect the top k most similar movies
    results = movie_data.iloc[indices].copy()
    results['similarity'] = similarity[indices]
    
    return results.sort_values(by='similarity', ascending=False)

# movies_data['genres_text'] = movies_data['genres'].apply(lambda x: ' '.join(x))


# vectorizer_genres = TfidfVectorizer(ngram_range=(1,2))
# tfidf_genres = vectorizer_genres.fit_transform(movies_data['genres'])

# def get_genres_by_title(title):
#     cleaned_title = re.sub("[^a-zA-Z0-9 ]", "", title).lower()
#     matched_row = movies_data[movies_data['title'].str.lower() == cleaned_title]
#     if not matched_row.empty:
#         return '|'.join(matched_row.iloc[0]['genres'])  # return genres as string
#     else:
#         return None

# def search_similar_genres_by_title(title):
#     genres = get_genres_by_title(title)
#     if genres is None:
#         return f"No genres found for title: {title}"

#     query_vec = vectorizer_genres.transform([genres])
#     similarity = cosine_similarity(query_vec, tfidf_genres).flatten()
#     indices = np.argpartition(similarity, -10)[-10:]
#     results = movies_data.iloc[indices].copy()
#     results['similarity'] = similarity[indices]
#     return results.sort_values(by='similarity', ascending=False)

# # Example usage
# print(search_similar_genres_by_title("Interstellar"))
