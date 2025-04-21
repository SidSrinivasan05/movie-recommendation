from flask import Flask, render_template, request
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

my_movie_catalog = pd.read_csv("movies_catalog.csv")

# Assuming movie data is stored in my_movie_catalog
def search_by_title(title, top_k=5, movie_data=my_movie_catalog):
    cleaned_title = re.sub("[^a-zA-Z0-9 ]", "", title).lower()
    vectorizer_title = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_title = vectorizer_title.fit_transform(movie_data['title'])
    query_vec = vectorizer_title.transform([cleaned_title])
    similarity = cosine_similarity(query_vec, tfidf_title).flatten()
    indices = np.argpartition(similarity, -top_k)[-top_k:]
    results = movie_data.iloc[indices].copy()
    results['similarity'] = similarity[indices]
    return results.sort_values(by='similarity', ascending=False)

# Web app route
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        if request.form.get("reset") == "1":
            results = None  # Clear results
        else:
            title = request.form.get("title", "")
            results = search_by_title(title)
    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)