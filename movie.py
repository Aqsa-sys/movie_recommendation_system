import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie data
movies = pd.read_csv("movies.csv")

# Convert genres to string format
movies["genres"] = movies["genres"].astype(str)

# Convert genre text into a matrix of token counts
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies["genres"])

# Compute cosine similarity between movies
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Function to get movie recommendations
def recommend_movie(title, num_recommendations=3):
    if title not in movies["title"].values:
        return f"Movie '{title}' not found in dataset."
    
    index = movies[movies["title"] == title].index[0]
    similarity_scores = list(enumerate(cosine_sim[index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    recommended_indices = [i[0] for i in sorted_scores[1:num_recommendations+1]]
    recommended_titles = movies["title"].iloc[recommended_indices]
    
    return recommended_titles.tolist()

# Example usage
print("Recommendations for 'The Matrix':")
print(recommend_movie("The Matrix"))
