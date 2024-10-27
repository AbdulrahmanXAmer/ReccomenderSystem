# Movie Recommendation System with Autoencoders and Feature Engineering

## Introduction

This project involves building a movie recommendation system using a combination of traditional machine learning, feature engineering, and deep learning techniques. The goal is to recommend movies based on learned embeddings from an autoencoder model. The notebook walks through the complete process, including data preprocessing, model building, and generating recommendations.

## Project Overview

The project aims to recommend movies that are similar based on features extracted from the dataset. The process involves:

1. **Loading and Exploring the Dataset**: Reading the movie dataset and exploring its structure.
2. **Handling Missing and Duplicate Values**: Cleaning the data by filling missing values and removing duplicates.
3. **Feature Engineering**: Creating new features based on existing columns such as title, description, stars, and genre.
4. **Encoding Categorical Features**: Using `LabelEncoder` for categorical features and multi-hot encoding for genres.
5. **Text Feature Representation**: Converting text data into numerical vectors using `TfidfVectorizer` and `HashingVectorizer`.
6. **Predicting Missing Certificates**: Using `RandomForestClassifier` to predict missing 'certificate' values.
7. **Multi-Target Regression for Ratings and Votes**: Using Ridge regression to predict missing ratings and votes.
8. **Autoencoder for Movie Embeddings**: Building an autoencoder to learn a low-dimensional embedding of the movie features.
9. **Cosine Similarity for Movie Recommendations**: Computing cosine similarity between the learned embeddings to find similar movies.
10. **Retrieving Similar Movies**: Creating a function to retrieve the top 10 most similar movies based on cosine similarity.

## Technologies Used

- **Python**: The primary programming language for building the application.
- **Pandas, NumPy, Scikit-Learn**: Used for data preprocessing, feature engineering, and model training.
- **TensorFlow**: Used for building and training the autoencoder.
- **RandomForestClassifier and Ridge Regression**: For classification and regression tasks.
- **Cosine Similarity**: For similarity-based recommendations.

## Dataset and Feature Engineering

- **Dataset**: The dataset (`n_movies.csv`) contains information about movies, including title, description, stars, genres, and other metadata.
- **Handling Missing Data**: Missing values in categorical columns are filled with placeholders, and missing ratings/votes are predicted using machine learning models.
- **Feature Creation**: Combining text-based columns such as title, description, stars, and genre into a `text_features` column to better represent movie characteristics.
- **Encoding**: Using TF-IDF and hashing vectorizers to convert text features into numerical representations.

## Model Building

- **RandomForestClassifier**: Used to predict missing 'certificate' values in the dataset.
- **Ridge Regression**: Used to predict missing `rating` and `votes` values for movies.
- **Autoencoder**: A neural network that learns low-dimensional embeddings for each movie, capturing the key features for similarity-based recommendations.

## Movie Recommendations

- **Embedding Extraction**: After training the autoencoder, the 16-dimensional embeddings for each movie are extracted.
- **Cosine Similarity**: These embeddings are used to compute the cosine similarity between movies, which allows us to recommend similar movies.
- **Recommendation Function**: A function is provided to retrieve the top 10 most similar movies based on cosine similarity.

## How to Run the Project

### Prerequisites

- **Python 3.x**: Ensure Python is installed.
- **Required Libraries**: Install the necessary libraries using the command below:

  ```sh
  pip install pandas numpy scikit-learn tensorflow
  ```

### Running the Notebook

1. **Clone the Repository**
   ```sh
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Run the Jupyter Notebook**
   Launch Jupyter Notebook and open the notebook file:
   ```sh
   jupyter notebook
   ```

3. **Execute the Cells**
   Run each cell sequentially to load data, preprocess, build models, and generate recommendations.



