## Netflix Movie Recommendation System with NLP: Content-Based Approach

### Description:
This repository houses a Netflix movie recommendation system built using natural language processing (NLP) techniques and a content-based approach. The system is designed to provide personalized movie recommendations based on user preferences and movie descriptions. It utilizes a dataset containing 5000 data points of movies.

### Key Details:
- **Dataset:** Netflix movie dataset with 5000 data points.
- **Tasks:** Checking null values, Cleaning Data, Utilizing NLP tasks.
- **Techniques Used:** Count Vectorizer for word vectorization, Cosine Similarity for measuring vector distances.
- **Recommendation Approach:** Content-based recommendation system leveraging NLP techniques.

### Project Overview:
The project begins with data preprocessing steps, including checking for null values and cleaning the data. It then employs various NLP tasks to analyze movie descriptions and extract relevant information. Using Count Vectorizer, the movie descriptions are converted into numerical vectors, allowing for similarity calculations using Cosine Similarity. This enables the recommendation system to suggest movies similar to those already liked by the user, based on content-based recommendation techniques.

### Repository Contents:
- **Jupyter Notebook:** Contains the Python code for data preprocessing, NLP tasks, and recommendation system implementation.
- **movie.py:** Contains code for deploying the recommendation system using FastAPI.
- **Dataset File:** Includes the Netflix movie dataset used for analysis.
- **Dataset Link:** [Netflix Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) - Please download the dataset from this Kaggle link before using it in the project.

### Instructions:

#### For Jupyter Notebook:
1. Clone this repository.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
  ### Instructions to run on vscode :  
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


uvicorn movie:app --host 0.0.0.0 --port 8001

