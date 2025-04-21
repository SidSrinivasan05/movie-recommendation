# Movie Recommendation System

This is a Flask-based movie recommendation system that helps users discover movies based on **title similarity**, **genre similarity**, or a combination of both. Built using TF-IDF vectorization and cosine similarity, it provides intelligent suggestions even when the input is not an exact match.

## 🔍 Features

- **Title-based Search**: Find movies with similar names using fuzzy text matching.
- **Genre-based Search**: Get recommendations based on genre similarity.
- **Hybrid Ranking**: Combines both title and genre similarities with configurable weights.
- **Fast and Lightweight**: Uses scikit-learn for vectorization and similarity calculations.
- **Simple Web UI**: Built with Flask for easy interaction.

## Tech Stack

- Python 3.9+
- Flask
- Pandas
- scikit-learn (TF-IDF & Cosine Similarity)
- HTML/CSS + JavaScript (for frontend interactions)

## Dataset

The project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/)


