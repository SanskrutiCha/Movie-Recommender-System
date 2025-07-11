import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ✅ TMDb API Key
TMDB_API_KEY = "84e9e418523e84f96e64d4c15329fc7f"

# =========================
# 🎨 Streamlit UI Styling
# =========================
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        max-width: 900px;
        margin: auto;
    }
    h1, h2 {
        color: #ff4b4b;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1.2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# 📦 Load model and data
# =========================
with open('svd_model.pkl', 'rb') as f:
    model = pickle.load(f)

movies = pd.read_csv('movies.csv')         # must have columns: movieId, title, genres
ratings = pd.read_csv('rat1.csv')          # must have columns: userId, movieId, rating

data = pd.merge(ratings, movies, on='movieId')
user_ids = sorted(data['userId'].unique())

# =========================
# 🔍 TF-IDF for content-based
# =========================
movies['genres'] = movies['genres'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(tfidf_matrix)

title_indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# =========================
# 🖼 Poster Fetching
# =========================
def get_poster(title):
    try:
        cleaned = title.split('(')[0].strip().replace(":", "").replace(",", "").replace("&", "and").replace("The ", "")
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={cleaned}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            results = response.json().get('results')
            if results:
                poster_path = results[0].get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        return None
    return None

# =========================
# 👤 User-Based Recommender
# =========================
def recommend_by_user(user_id, n=5):
    seen = data[data['userId'] == user_id]['movieId'].tolist()
    unseen = list(set(movies['movieId']) - set(seen))

    predictions = [(mid, model.predict(user_id, mid).est) for mid in unseen]
    top = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

    result = []
    for mid, _ in top:
        title = movies[movies['movieId'] == mid]['title'].values[0]
        poster = get_poster(title)
        result.append((title, poster))
    return result

# =========================
# 🎬 Title-Based Recommender
# =========================
def recommend_by_title(title, n=5):
    if title not in title_indices:
        return []
    idx = title_indices[title]
    distances, indices = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=n+1)
    
    results = []
    for i in indices[0][1:]:
        t = movies.iloc[i]['title']
        poster = get_poster(t)
        results.append((t, poster))
    return results

# =========================
# 🌐 Streamlit Interface
# =========================
st.title("🍿 Movie Recommender")
st.markdown("🎯 Get movie recommendations based on your user ID or movie title.")

mode = st.radio("Select mode:", ["🔑 Recommend by User ID", "🔍 Search by Movie Title"])

if mode == "🔑 Recommend by User ID":
    selected_user = st.selectbox("Select user ID", user_ids)
    if st.button("🎥 Recommend"):
        recs = recommend_by_user(selected_user)
        if recs:
            cols = st.columns(2)
            for i, (title, poster) in enumerate(recs):
                with cols[i % 2]:
                    st.subheader(title)
                    if poster:
                        st.image(poster, width=250)
                    else:
                        st.text("Poster not available.")
        else:
            st.warning("No recommendations found.")

else:
    input_title = st.text_input("Enter full movie title (e.g. Toy Story (1995))")
    if st.button("🔎 Recommend"):
        recs = recommend_by_title(input_title)
        if recs:
            cols = st.columns(2)
            for i, (title, poster) in enumerate(recs):
                with cols[i % 2]:
                    st.subheader(title)
                    if poster:
                        st.image(poster, width=250)
                    else:
                        st.text("Poster not available.")
        else:
            st.warning("Movie not found or no similar movies.")
