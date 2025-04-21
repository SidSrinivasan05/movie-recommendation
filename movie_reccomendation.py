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

genre_movies.to_csv("movies_title_catalog.csv", index=False)

def search_by_title(title, top_k=10, movie_data=genre_movies):
    # Clean the title input
    cleaned_title = re.sub("[^a-zA-Z0-9 ]", "", title).lower()
    
    vectorizer_title = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_title = vectorizer_title.fit_transform(movie_data['title'])
    
    query_vec = vectorizer_title.transform([cleaned_title])     # Transform the query title into its tf-idf vector representation
    similarity = cosine_similarity(query_vec, tfidf_title).flatten() # Compute cosine similarity between the input title and all movie titles
    
    indices = np.argpartition(similarity, -top_k)[-top_k:]    # Get the indices of the top k most similar movies
    
    results = movie_data.iloc[indices].copy()
    results['similarity'] = similarity[indices]
    
    return results.sort_values(by='similarity', ascending=False)

genre_catalog = pd.read_csv("genres_catalog.csv")
extra_catalog = pd.read_csv("movies.csv")
extra_catalog['genres'] = extra_catalog['genres'].str.split('|')


vectorizer_genres = TfidfVectorizer(ngram_range=(1,2))
tfidf_genres = vectorizer_genres.fit_transform(genre_catalog['genres_text'])

def search_by_genres(title, top_k=10, movie_data=genre_catalog):
    gen = get_genres(title)
    if gen is None:
        return None
    genres = ' '.join( list(gen) )
    query_vec = vectorizer_genres.transform([genres])
    
    similarity = cosine_similarity(query_vec, tfidf_genres).flatten()
    indices = np.argpartition(similarity, -top_k)[-top_k:]
    
    results = movie_data.iloc[indices].copy()
    results['similarity'] = similarity[indices]
    return results.sort_values(by='similarity', ascending=False)

def get_genres(word):
    t_catalog = pd.DataFrame(extra_catalog['genres'].values, index=extra_catalog['title']).T
    for col in t_catalog.columns:
        if word.lower() in col.lower():
            return list(t_catalog[col])[0]
        
        
def combine_searches(movie, weight_title=0.7, weight_genre=0.3):
    title_df = search_by_title(movie)
    genre_df = search_by_genres(movie)
    if genre_df is None:
        return title_df
    if title_df is None:
        return genre_df
    # Normalize similarity scores between 0 and 1
    title_df['similarity'] = title_df['similarity'] / title_df['similarity'].max()
    genre_df['similarity'] = genre_df['similarity'] / genre_df['similarity'].max()

    # Rename for clarity
    title_df.rename(columns={'similarity': 'title_score'}, inplace=True)
    genre_df.rename(columns={'similarity': 'genre_score'}, inplace=True)

    # Merge on title (inner or outer depending on if you want partial matches)
    combined = pd.merge(title_df, genre_df, on='title', how='outer')

    combined['title_score'] = combined['title_score'].fillna(0)
    combined['genre_score'] = combined['genre_score'].fillna(0)

    # Weighted similarity
    combined['similarity'] = (
        weight_title * combined['title_score'] +
        weight_genre * combined['genre_score']
    )

    return combined.sort_values(by='similarity', ascending=False)

title = "Pirates of the Carribean"

# print(genre_movies.head())
# print(extra_catalog.head())

print(search_by_title(title))
print(search_by_genres(title))

print(combine_searches(title))