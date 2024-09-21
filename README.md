# Movie Recommendation System
## Project Overview
This project recommend movies to users based on their preferences or past viewing history. It employs TF-IDF Vectorization and Cosine Similarity to provide content-based recommendations by analyzing movie overviews and genres.

## Features
* Data Preprocessing: The dataset is loaded and cleaned. Missing overviews are filled with empty strings, and genres are extracted and processed into a readable format.
* Feature Engineering: A combined feature is created by merging movie overviews and genres, which is used for recommendations.
* Recommendation Engine:
 TF-IDF Vectorizer converts movie features into a numerical form.
 Cosine Similarity calculates similarity between movies to generate recommendations based on user input.
* Visualization: The project includes a histogram that shows the distribution of movie genres.


## Tech Stack
* Programming Language: Python
* Libraries:
   pandas and numpy for data manipulation
   matplotlib for visualizations
   scikit-learn for feature extraction (TF-IDF) and similarity calculations (Cosine Similarity)
   ast for handling JSON data in the dataset
* Dataset: TMDB 5000 Movies Dataset


