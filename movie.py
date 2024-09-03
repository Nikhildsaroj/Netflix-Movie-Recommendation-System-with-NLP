from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.responses import HTMLResponse
import requests

app = FastAPI()

# Load the model components from the pickle file
with open('movie_recommender_model.pkl', 'rb') as file:
    model_components = pickle.load(file)

# Extract components from the loaded model
new_df = model_components['new_df']
vectors = model_components['vectors']
similarity = model_components['similarity']

# TMDB API details
TMDB_API_KEY = '88f402126ce96431d1bb56587cea4458'
TMDB_API_URL = 'https://api.themoviedb.org/3'

# Define a class for the recommendation request
class RecommendationRequest(BaseModel):
    movie_title: str

# Define a class for the recommendation response
class RecommendationResponse(BaseModel):
    movie_title: str
    recommendations: List[dict]

def get_top_indices(similarity_matrix, vector_index, top_n=6):
    similarity_scores = similarity_matrix[vector_index]
    sorted_indices = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in sorted_indices[:top_n]]
    return top_indices

def recommend_movies(movie_title, new_df, similarity_matrix, top_n=5):
    try:
        movie_index = new_df[new_df['title'] == movie_title].index[0]
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Movie '{movie_title}' not found in the database.")

    top_indices = get_top_indices(similarity_matrix, movie_index, top_n+1)[1:]
    recommended_movies = new_df.iloc[top_indices]
    return recommended_movies

def search_movie_by_title(title):
    response = requests.get(f"{TMDB_API_URL}/search/movie", params={
        'api_key': TMDB_API_KEY,
        'query': title,
        'language': 'en-US'
    })
    data = response.json()
    return data['results'][0] if data['results'] else None

def fetch_movie_details(title):
    movie = search_movie_by_title(title)
    if movie:
        movie_id = movie['id']
        try:
            # Fetch movie details
            details_response = requests.get(f"{TMDB_API_URL}/movie/{movie_id}", params={
                'api_key': TMDB_API_KEY,
                'language': 'en-US'
            })
            details_response.raise_for_status()
            details = details_response.json()
            
            # Fetch movie videos (trailers)
            videos_response = requests.get(f"{TMDB_API_URL}/movie/{movie_id}/videos", params={
                'api_key': TMDB_API_KEY,
                'language': 'en-US'
            })
            videos_response.raise_for_status()
            videos = videos_response.json()
            
            # Extract trailer information
            trailers = [video for video in videos.get('results', []) if video['type'] == 'Trailer']
            trailer_url = None
            if trailers:
                trailer_key = trailers[0]['key']
                trailer_url = f"https://www.youtube.com/embed/{trailer_key}"
            
            return {
                'title': details.get('title'),
                'poster_url': f"https://image.tmdb.org/t/p/w500{details.get('poster_path', '')}",
                'overview': details.get('overview'),
                'trailer_url': trailer_url
            }
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail="Error fetching movie details from TMDB.")
    
    raise HTTPException(status_code=404, detail="Movie details not found")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
   <!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Movie Recommender System</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f4;
            color: #000;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 1100px;
            margin: 40px auto;
        }

        .welcome-section {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            margin-bottom: 40px;
        }

        #recommendation-form {
            margin: 20px 0;
        }

        #recommendation-form input[type="text"] {
            margin: 10px 0;
            background-color: white;
            color: #000;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 10px;
        }

        #recommendation-form button {
            background-color: #e50914;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            font-size: 16px;
            font-weight: bold;
            color: white;
        }

        #result {
            margin-top: 20px;
        }

        .movie {
            display: flex;
            height: 50%;
            flex-direction: row;
            margin-bottom: 30px;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .movie:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        }

        .movie-poster {
            max-width: 40%;
            height: 500px;
            margin-right: 20px;
            border-radius: 8px;
        }

        .movie-details {
            text-align: left;
        }

        .movie-details h2 {
            margin-top: 0;
            font-size: 24px;
        }

        .movie-details p {
            font-size: 16px;
        }

        .trailer-video {
            margin-top: 20px;
        }

        iframe {
            width: 100%;
            height: 350px;
            border-radius: 8px;
        }

        footer {
            margin-top: 40px;
            padding: 20px;
            background-color: white;
            border-top: 1px solid #333;
            text-align: center;
        }

        .spinner {
            display: none;
            margin: 20px auto;
            border: 8px solid #f3f3f3;
            border-top: 8px solid #333;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% {
                transform: rotate(0deg);
            }

            100% {
                transform: rotate(360deg);
            }
        }

        @media (max-width: 768px) {
            .movie {
                flex-direction: column;
                align-items: center;
            }

            .movie-poster {
                max-width: 40%;
                height: 20%;
                margin-right: 0;
            }

            .movie-details {
                text-align: center;
            }
        }
    </style>
</head>

<body>
    <div class="container">
        <div class="welcome-section">
            <h1>Welcome to the Movie Recommender System!</h1>
            <p>Enter the title of a movie to receive a list of recommended movies similar to your input.</p>
        </div>
        <form id="recommendation-form">
            <h3>Get Movie Recommendations</h3>
            <input type="text" id="movie-title" name="movie_title" class="form-control" placeholder="Enter movie title">
            <button type="submit" class="btn btn-primary mt-2">Get Recommendations</button>
        </form>
        <div id="loading-spinner" class="spinner"></div>
        <div id="result"></div>
        <footer>
            <p>Made with ❤️ by Nikhil Saroj</p>
        </footer>
    </div>
    <script>
        document.getElementById('recommendation-form').addEventListener('submit', async function (event) {
            event.preventDefault();
            const movieTitle = document.getElementById('movie-title').value;
            document.getElementById('loading-spinner').style.display = 'block';
            const response = await fetch('/recommend-movies/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ movie_title: movieTitle })
            });
            document.getElementById('loading-spinner').style.display = 'none';
            const result = await response.json();
            if (result.recommendations) {
                const recommendationsHtml = result.recommendations.map(movie => `
                    <div class="movie">
                        <img src="${movie.poster_url}" class="movie-poster" alt="${movie.title}">
                        <div class="movie-details">
                            <h1>${movie.title}</h1>
                            <p>${movie.overview}</p>
                            ${movie.trailer_url ? `<iframe class="trailer-video" src="${movie.trailer_url}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>` : '<p>No trailer available</p>'}
                        </div>
                    </div>
                `).join('');
                document.getElementById('result').innerHTML = `
                    <h3>Recommendations for '${result.movie_title}':</h3>
                    ${recommendationsHtml}
                `;
            } else {
                document.getElementById('result').innerText = 'Error: ' + result.detail;
            }
        });
    </script>
</body>

</html>



    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/recommend-movies/", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    movie_title = request.movie_title
    try:
        recommendations = recommend_movies(movie_title, new_df, similarity, top_n=10)
        result = []
        for _, row in recommendations.iterrows():
            # Search for movie details on TMDB
            movie_details = fetch_movie_details(row['title'])
            result.append({
                "title": movie_details.get('title', 'No title available'),
                "poster_url": movie_details.get('poster_url', 'No poster available'),
                "overview": movie_details.get('overview', 'No overview available'),
                "trailer_url": movie_details.get('trailer_url', '')
            })
        
        return {"movie_title": movie_title, "recommendations": result}
    except HTTPException as e:
        raise e
