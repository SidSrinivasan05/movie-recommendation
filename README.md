# Movie Recommendation System

This is a Flask-based movie recommendation system that helps users discover movies based on **title similarity**, **genre similarity**, or a combination of both. Built using TF-IDF vectorization and cosine similarity, it provides intelligent suggestions even when the input is not an exact match.

## 🔍 Features

- **Title-based Search**: Find movies with similar names using fuzzy text matching.
- **Genre-based Search**: Get recommendations based on genre similarity.
- **Hybrid Ranking**: Combines both title and genre similarities with configurable weights.
- **Fast and Lightweight**: Uses scikit-learn for vectorization and similarity calculations.
- **Simple Web UI**: Built with Flask for easy interaction.

## Tech Stack

- Python
- Flask
- Pandas
- scikit-learn (TF-IDF & Cosine Similarity)
- HTML/CSS + JavaScript (for frontend interactions)

## Dataset

The project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/)


## Usage

To see the web interface and use the movie reccomender for yourself, first clone the repo. Then run

```
python app.py
```

and open the following in your browser

```
http://127.0.0.1:5000/
```
