import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

# Load preprocessed data
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Drop timestamp
ratings = ratings.drop(columns=['timestamp'], errors='ignore')

# User-Item Matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Normalize by centering
user_ratings_mean = np.mean(user_item_matrix.values, axis=1)
R_demeaned = user_item_matrix.values - user_ratings_mean.reshape(-1, 1)

# Apply SVD
U, sigma, Vt = svds(R_demeaned, k=50)
sigma = np.diag(sigma)
preds = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

norm_preds = pd.DataFrame(preds, columns=user_item_matrix.columns, index=user_item_matrix.index)

# Compute TF-IDF cosine similarity for content-based filtering
movie_features = movies.copy()
movie_features['genres'] = movie_features['genres'].fillna('')
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movie_features['genres'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=movies['movieId'], columns=movies['movieId'])

# Ensure index types match
cosine_sim_df.index = cosine_sim_df.index.astype(int)
cosine_sim_df.columns = cosine_sim_df.columns.astype(int)
norm_preds.index = norm_preds.index.astype(int)
norm_preds.columns = norm_preds.columns.astype(int)

# Recommendation function
def get_top_hybrid_recommendations(user_id, norm_preds, cosine_sim, ratings_df, movies_df, alpha=0.5, top_n=10):
    cosine_sim.index = cosine_sim.index.astype(int)
    cosine_sim.columns = cosine_sim.columns.astype(int)

    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    user_rated_movie_ids = set(user_ratings['movieId'])

    user_collab_scores = norm_preds.loc[user_id].drop(index=user_rated_movie_ids, errors='ignore')

    user_high_rated = user_ratings[user_ratings['rating'] >= 4]['movieId'].astype(int).tolist()

    content_scores = {}
    for movie_id in cosine_sim.index:
        if movie_id in user_rated_movie_ids:
            continue
        try:
            sim_scores = cosine_sim.loc[movie_id, user_high_rated].mean()
            content_scores[movie_id] = sim_scores
        except KeyError:
            continue

    content_scores_series = pd.Series(content_scores)
    user_collab_scores = user_collab_scores.reindex(content_scores_series.index).fillna(0)
    content_scores_series = content_scores_series.fillna(0)

    hybrid_scores = alpha * user_collab_scores + (1 - alpha) * content_scores_series
    top_movie_ids = hybrid_scores.sort_values(ascending=False).head(top_n).index

    recommendations = movies_df[movies_df['movieId'].isin(top_movie_ids)].copy()
    recommendations['hybrid_score'] = recommendations['movieId'].map(hybrid_scores)
    recommendations = recommendations.sort_values(by='hybrid_score', ascending=False)

    return recommendations[['title', 'hybrid_score']]

# Streamlit UI
st.title("🎬 Hybrid Movie Recommendation System")

user_id = st.number_input("Enter User ID:", min_value=int(ratings['userId'].min()), max_value=int(ratings['userId'].max()), value=1)
top_n = st.slider("Number of Recommendations:", 1, 20, 10)
alpha = st.slider("Weight for Collaborative Filtering (0 = Content only, 1 = Collaborative only):", 0.0, 1.0, 0.7)

if st.button("Get Recommendations"):
    try:
        recommendations = get_top_hybrid_recommendations(user_id, norm_preds, cosine_sim_df, ratings, movies, alpha, top_n)
        st.subheader("Recommended Movies:")
        st.table(recommendations.reset_index(drop=True))
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
