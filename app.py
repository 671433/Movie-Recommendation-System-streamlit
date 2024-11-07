import gc
import os
import gdown
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import pickle
import streamlit as st

API_KEY = '3acc39754e9014eca3c0799f663b5785'
BASE_URL = 'https://api.themoviedb.org/3'

data_loaded = False
df = None
cosine_sim = None
nb_sentiment_model = None
svm_sentiment_model = None
vectorizer = None
vectorizer_svm = None

def load_data():
    global df, cosine_sim, nb_sentiment_model, svm_sentiment_model, vectorizer, vectorizer_svm, data_loaded

    if not data_loaded:
        # URL of the file in Google Drive (for the dataset)
        file_url_movies = 'https://drive.google.com/uc?id=18gEjCisVUR5GuFCgyuCR_FFG2LBJDbvR&export=download'

        # URL of the file for the svm_sentiment_model.pkl
        file_url_svm_model = 'https://drive.google.com/uc?id=1j7O1sc9k1Law5vYRLZzjTxQL3VrNpEVZ&export=download'

        # Use gdown to download the movies dataset
        gdown.download(file_url_movies, 'new_movies.csv', quiet=False)

        # Download the svm sentiment model
        gdown.download(file_url_svm_model, 'svm_sentiment_model.pkl', quiet=False)

        # Load the DataFrame
        dtype_dict = {
            'id': 'int32',
            'title': 'str',
            'concat': 'str',
        }

        df = pd.read_csv('new_movies.csv', low_memory=False, encoding='utf-8', nrows=10000, dtype=dtype_dict, usecols=['id', 'title', 'concat'])

        # Clean 'concat' column (remove NaN values)
        df = df[df['concat'].notna()]

        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(analyzer='word', stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df['concat'])

        # Initialize cosine similarity matrix
        chunk_size = 500
        n_chunks = len(df) // chunk_size + 1
        cosine_sim = np.zeros((len(df), len(df)), dtype='float32')

        # Loop through the data in chunks
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(df))
            chunk = tfidf_matrix[start:end]

            # Skip empty chunks
            if chunk.shape[0] == 0:
                print(f"Warning: Empty chunk at index {i}, skipping...")
                continue

            try:
                # Compute cosine similarity for this chunk
                cosine_sim[start:end] = cosine_similarity(chunk, tfidf_matrix)
            except ValueError as e:
                print(f"Error in computing cosine similarity for chunk {i}: {e}")
                continue

        # Load models
        with open('nb_sentiment_model.pkl', 'rb') as f:
            nb_sentiment_model = pickle.load(f)
        with open('review_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('svm_sentiment_model.pkl', 'rb') as f:
            svm_sentiment_model = pickle.load(f)
        with open('review_vectorizer_SVM.pkl', 'rb') as f:
            vectorizer_svm = pickle.load(f)

        # Clear unnecessary objects to free memory
        gc.collect()

        # Set the flag to True to indicate data has been loaded
        data_loaded = True

def analyze_review_sentiment(review_text):
    try:
        # Transform the review text using the vectorizer
        review_vector = vectorizer.transform([review_text])
        # Predict sentiment
        prediction = nb_sentiment_model.predict(review_vector)[0]
        return 'POSITIVE' if prediction == 1 else 'NEGATIVE'
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 'UNKNOWN'

def analyze_review_sentiment_by_svm(review_text):
    try:
        # Transform the review text using the vectorizer
        review_vector = vectorizer_svm.transform([review_text])
        # Predict sentiment
        prediction = svm_sentiment_model.predict(review_vector)[0]
        return 'POSITIVE' if prediction == 1 else 'NEGATIVE'
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 'UNKNOWN'

def get_recommendations(title):
    if title not in df['title'].values:
        return 'Sorry! try another movie name'

    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # higher 5 results
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = []

    for movie_index in movie_indices:
        movie_id = df['title'].iloc[movie_index]
        movie_details = get_movie_details(movie_id)

        if movie_details:
            # Round the rating to one decimal place
            vote_average = movie_details.get('vote_average', 0)
            rounded_vote_average = round(vote_average, 1) if isinstance(vote_average, (int, float)) else vote_average

            recommended_movies.append({
                'title': movie_details.get('title', ''),
                "poster_path": movie_details.get('poster_path', ''),
                "overview": movie_details.get('overview', ''),
                "rating": movie_details.get('rating', ''),
                "release_date": movie_details.get('release_date', ''),
                "vote_average": rounded_vote_average,
                "vote_count": movie_details.get('vote_count', ''),
            })
    return recommended_movies

def get_movie_details(movie_name):
    url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={movie_name}"
    print(f"Fetching URL: {url}")
    response = requests.get(url)
    print(f"Response Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            movie_details = data['results'][0]
            print(f"Movie details: {movie_details}")
            return movie_details
    print("No results found.")
    return None

def get_movie_reviews(movie_id):
    # Get movie reviews and analyze their sentiment
    url = f"{BASE_URL}/movie/{movie_id}/reviews?api_key={API_KEY}"
    print(f"Fetching Reviews URL: {url}")
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        reviews = data.get('results', [])

        # Check if there are reviews
        if not reviews:
            return []

        # Analyze sentiment for each review
        for review in reviews:
            # Extract the review content
            review_text = review.get('content', '')
            # Add sentiment analysis to each review
            review['sentiment'] = analyze_review_sentiment_by_svm(review_text)

            # Confidence score
            review['confidence'] = 0.8 if len(review_text) > 100 else 0.5

        print(f"Processed {len(reviews)} reviews with sentiment analysis")
        return reviews

    print("No reviews found.")
    return []


def main():
    load_data()

    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem;
            background: var(--secondary-background-color);
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .movie-container {
            display: flex;
            gap: 2rem;
            margin: 2rem 0;
            padding: 1rem;
            background: var(--secondary-background-color);
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .reviews-intro {
            background: var(--secondary-background-color);
            padding: 1.5rem;
            border-radius: 10px;
            margin: 2rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .review-card {
            background: var(--secondary-background-color);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid;
        }
        .review-card.positive {
            border-left-color: #28a745;
        }
        .review-card.negative {
            border-left-color: #dc3545;
        }
        .movie-details, .review-card, .reviews-intro {
            color: var(--text-color);
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("<h1 class='main-header'>Movie Recommendation System</h1>", unsafe_allow_html=True)

    # Introduction
    st.markdown("""
        <div class="reviews-intro">
            <h3>Discover Your Next Favorite Movie:</h3>
            <p>
                Welcome to my movie recommendation platform! Just enter a movie title to see similar films,
                complete with details like summaries, release dates, and ratings pulled from a trusted API.
                My platform also analyzes reviews to predict sentiment, making it easy to gauge each film's reception.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Search form
    movie_title = st.text_input("Enter a movie title:", placeholder="Search for a movie...")

    if st.button("Get Recommendations"):
        # Get movie details
        movie_details = get_movie_details(movie_title)

        if movie_details:
            # Display movie details in two columns
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(f"https://image.tmdb.org/t/p/w500/{movie_details['poster_path']}")

            with col2:
                st.markdown(f"""
                    <div class="movie-details">
                        <h2>{movie_details['title']}</h2>
                        <p><strong>Overview:</strong> {movie_details['overview']}</p>
                        <p><strong>Release Date:</strong> {movie_details['release_date']}</p>
                        <p><strong>Rating:</strong> {round(movie_details['vote_average'], 1)}/10</p>
                        <p><strong>Vote Count:</strong> {movie_details['vote_count']}</p>
                    </div>
                """, unsafe_allow_html=True)

            # Reviews Section
            st.markdown("""
                <div class="reviews-intro">
                    <h3>Reviews Analysis</h3>
                    <p>
                        Reviews are analyzed using a Support Vector Machine model, 
                        providing accurate sentiment classification for both short and long texts.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            movie_id = movie_details['id']
            reviews = get_movie_reviews(movie_id)

            if reviews:
                for review in reviews:
                    sentiment_class = "positive" if review['sentiment'] == 'POSITIVE' else "negative"
                    st.markdown(f"""
                        <div class="review-card {sentiment_class}">
                            <h4>{review.get('author', 'Anonymous')}</h4>
                            <p><strong>Sentiment:</strong> {review['sentiment']}</p>
                            <p>{review['content']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No reviews available for this movie.")

            # Recommendations Section
            st.markdown("""
                <div class="reviews-intro">
                    <h3>Recommended Movies</h3>
                    <p>
                        Recommendations are based on content similarity, analyzing genres, cast, 
                        director, and descriptions using TF-IDF and cosine similarity.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            recommendations = get_recommendations(movie_title)
            if isinstance(recommendations, str):
                st.warning(recommendations)
            else:
                # Display recommendations in a grid
                cols = st.columns(3)
                for idx, movie in enumerate(recommendations):
                    with cols[idx % 3]:
                        st.image(f"https://image.tmdb.org/t/p/w500/{movie['poster_path']}")
                        st.markdown(f"""
                            <div style='text-align: center;'>
                                <h4>{movie['title']}</h4>
                                <p>Rating: {movie['vote_average']}/10</p>
                            </div>
                        """, unsafe_allow_html=True)
                        with st.expander("More Details"):
                            st.write(f"Overview: {movie['overview']}")
                            st.write(f"Release Date: {movie['release_date']}")
                            st.write(f"Vote Count: {movie['vote_count']}")

    # Footer
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <p>Powered By Tamer Al Deen _ DAT152 _ ML03</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()