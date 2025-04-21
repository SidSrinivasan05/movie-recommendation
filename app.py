from flask import Flask, render_template, request
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

title_catalog = pd.read_csv("movies_title_catalog.csv")
genre_catalog = pd.read_csv("genres_catalog.csv")

# Assuming movie data is stored in my_movie_catalog
def search_by_title(title, top_k=5, movie_data=title_catalog):
    cleaned_title = re.sub("[^a-zA-Z0-9 ]", "", title).lower()
    vectorizer_title = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_title = vectorizer_title.fit_transform(movie_data['title'])
    query_vec = vectorizer_title.transform([cleaned_title])
    similarity = cosine_similarity(query_vec, tfidf_title).flatten()
    indices = np.argpartition(similarity, -top_k)[-top_k:]
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

def combine_searches(movie):
    genre_listing = search_by_genres(movie)
    title_listing = search_by_title(movie)
    
    return pd.concat([genre_listing, title_listing], axis=0).sort_values(by='similarity', ascending=False)


# Web app route
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        if request.form.get("reset") == "1":
            results = None  # Clear results
        else:
            title = request.form.get("title", "")
            results = combine_searches(title)
    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)