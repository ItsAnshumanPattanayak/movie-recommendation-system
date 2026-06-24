# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import MovieLensLoader
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFiltering
from src.matrix_factorization import MatrixFactorization
from src.evaluation import RecommenderEvaluator

# ==================== Page Configuration ====================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS ====================

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        border: none;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .movie-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #FF4B4B;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .movie-title {
        font-size: 18px;
        font-weight: bold;
        color: #1f1f1f;
        margin-bottom: 5px;
    }
    .movie-genre {
        font-size: 14px;
        color: #666;
        font-style: italic;
    }
    .movie-score {
        font-size: 16px;
        color: #FF4B4B;
        font-weight: bold;
    }
    h1 {
        color: #1f1f1f;
        font-weight: 800;
    }
    h2 {
        color: #FF4B4B;
        font-weight: 700;
    }
    h3 {
        color: #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== Session State Initialization ====================

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'loader' not in st.session_state:
    st.session_state.loader = None
if 'cb_model' not in st.session_state:
    st.session_state.cb_model = None
if 'cf_model' not in st.session_state:
    st.session_state.cf_model = None
if 'mf_model' not in st.session_state:
    st.session_state.mf_model = None

# ==================== Caching Functions ====================

@st.cache_resource(show_spinner=False)
def load_data(data_path):
    """Load and cache data"""
    loader = MovieLensLoader(data_path)
    if loader.load_data(verbose=False):
        loader.create_user_item_matrix()
        return loader
    return None

@st.cache_resource(show_spinner=False)
def train_content_based(_loader):
    """Train and cache content-based recommender"""
    recommender = ContentBasedRecommender(
        _loader.movies,
        _loader.ratings,
        _loader.tags
    )
    recommender.compute_similarity()
    return recommender

@st.cache_resource(show_spinner=False)
def train_collaborative(_loader):
    """Train and cache collaborative filtering"""
    cf = CollaborativeFiltering(_loader.ratings, _loader.movies)
    cf.compute_user_similarity()
    cf.compute_item_similarity()
    return cf

@st.cache_resource(show_spinner=False)
def train_matrix_factorization(_loader, n_factors=50, n_epochs=10):
    """Train and cache matrix factorization"""
    mf = MatrixFactorization(_loader.ratings, _loader.movies)
    mf.train(n_factors=n_factors, n_epochs=n_epochs, verbose=False)
    return mf

# ==================== Helper Functions ====================

def display_movie_card(row, show_score=True, score_label="Score"):
    """Display a movie as a card"""
    st.markdown(f"""
        <div class="movie-card">
            <div class="movie-title">🎬 {row['title']}</div>
            <div class="movie-genre">📁 {row['genres']}</div>
            {f'<div class="movie-score">⭐ {score_label}: {row.get("score", row.get("predicted_rating", row.get("similarity_score", 0))):.2f}</div>' if show_score else ''}
        </div>
    """, unsafe_allow_html=True)

def display_metric_card(title, value, icon="📊"):
    """Display a metric card"""
    st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; color: white;">{icon}</h3>
            <h2 style="margin:5px 0; color: white;">{value}</h2>
            <p style="margin:0; color: rgba(255,255,255,0.8);">{title}</p>
        </div>
    """, unsafe_allow_html=True)

# ==================== Main Application ====================

def main():
    # ==================== Sidebar ====================
    
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/movie.png", width=80)
        st.title("🎬 Movie Recommender")
        st.markdown("---")
        
        # Data loading section
        st.subheader("📁 Data Configuration")
        
        data_path = st.text_input(
            "Dataset Path",
            value="dataset",
            help="Path to folder containing MovieLens CSV files"
        )
        
        if st.button("🔄 Load Data", use_container_width=True):
            with st.spinner("Loading data..."):
                loader = load_data(data_path)
                if loader:
                    st.session_state.loader = loader
                    st.session_state.data_loaded = True
                    st.success("✅ Data loaded successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Error loading data. Check the path.")
        
        # Navigation
        if st.session_state.data_loaded:
            st.markdown("---")
            st.subheader("📊 Navigation")
            
            page = st.radio(
                "Go to",
                [
                    "🏠 Home",
                    "📊 Data Explorer",
                    "🎯 Content-Based",
                    "👥 Collaborative Filtering",
                    "🔢 Matrix Factorization",
                    "⚖️ Compare Models",
                    "📈 Evaluation"
                ],
                label_visibility="collapsed"
            )
            
            # Info
            st.markdown("---")
            st.subheader("ℹ️ About")
            st.info("""
                This system implements multiple 
                recommendation algorithms:
                
                - **Content-Based**: Genre & tag similarity
                - **Collaborative**: User & item patterns
                - **Matrix Factorization**: SVD algorithm
            """)
            
            # Stats
            if st.session_state.loader:
                st.markdown("---")
                st.subheader("📈 Quick Stats")
                loader = st.session_state.loader
                st.metric("Users", f"{loader.ratings['userId'].nunique():,}")
                st.metric("Movies", f"{len(loader.movies):,}")
                st.metric("Ratings", f"{len(loader.ratings):,}")
    
    # ==================== Main Content ====================
    
    if not st.session_state.data_loaded:
        show_welcome_page()
    else:
        loader = st.session_state.loader
        page = st.session_state.get('page', '🏠 Home')
        
        # Route to appropriate page
        if "Home" in page:
            show_home(loader)
        elif "Data Explorer" in page:
            show_data_explorer(loader)
        elif "Content-Based" in page:
            show_content_based(loader)
        elif "Collaborative" in page:
            show_collaborative(loader)
        elif "Matrix Factorization" in page:
            show_matrix_factorization(loader)
        elif "Compare" in page:
            show_comparison(loader)
        elif "Evaluation" in page:
            show_evaluation(loader)

# ==================== Page Functions ====================

def show_welcome_page():
    """Welcome page when data is not loaded"""
    st.title("🎬 Movie Recommendation System")
    st.markdown("### Welcome! Let's get started")
    
    st.info("👈 **Please load the dataset from the sidebar to begin**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("---")
        st.markdown("### 📋 Quick Start Guide")
        
        st.markdown("""
        1. **Download the MovieLens dataset**
           - Visit [GroupLens](https://grouplens.org/datasets/movielens/)
           - Download the **MovieLens Latest Small** dataset
           - Extract to a folder (e.g., `dataset/`)
        
        2. **Load the data**
           - Enter the folder path in the sidebar
           - Click "Load Data"
           - Wait for confirmation
        
        3. **Explore & Recommend**
           - Browse the data explorer
           - Try different recommendation algorithms
           - Compare results
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Features")
        
        feature_col1, feature_col2, feature_col3 = st.columns(3)
        
        with feature_col1:
            st.markdown("""
            #### Content-Based
            - Genre similarity
            - Tag analysis
            - TF-IDF vectorization
            - Cosine similarity
            """)
        
        with feature_col2:
            st.markdown("""
            #### Collaborative
            - User-User filtering
            - Item-Item filtering
            - Neighborhood methods
            - Pattern recognition
            """)
        
        with feature_col3:
            st.markdown("""
            #### Matrix Factorization
            - SVD algorithm
            - Latent features
            - Rating prediction
            - Cross-validation
            """)
    
    with col2:
        st.markdown("---")
        st.markdown("### 📊 Technologies")
        
        st.markdown("""
        - Python 3.8+
        - Pandas & NumPy
        - Scikit-learn
        - Surprise
        - Streamlit
        - Plotly
        """)
        
        st.markdown("---")
        st.markdown("### 🔗 Links")
        
        st.markdown("""
        - [GitHub Repo](#)
        - [Documentation](#)
        - [Report Issues](#)
        """)

def show_home(loader):
    """Home dashboard page"""
    st.title("🎬 Movie Recommendation System")
    st.markdown("### Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric_card(
            "Total Users",
            f"{loader.ratings['userId'].nunique():,}",
            "👥"
        )
    
    with col2:
        display_metric_card(
            "Total Movies",
            f"{len(loader.movies):,}",
            "🎬"
        )
    
    with col3:
        display_metric_card(
            "Total Ratings",
            f"{len(loader.ratings):,}",
            "⭐"
        )
    
    with col4:
        avg_rating = loader.ratings['rating'].mean()
        display_metric_card(
            "Average Rating",
            f"{avg_rating:.2f}",
            "📊"
        )
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Rating Distribution")
        rating_counts = loader.ratings['rating'].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            labels={'x': 'Rating', 'y': 'Count'},
            title="How users rate movies",
            color=rating_counts.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎭 Top Genres")
        # Extract genres
        genres = loader.movies['genres'].str.split('|', expand=True).stack()
        genre_counts = genres.value_counts().head(10)
        
        fig = px.bar(
            y=genre_counts.index,
            x=genre_counts.values,
            orientation='h',
            labels={'x': 'Count', 'y': 'Genre'},
            title="Most common genres",
            color=genre_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Popular movies
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Most Rated Movies")
        popular = loader.get_popular_movies(n=10)
        
        for idx, row in popular.iterrows():
            with st.container():
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    st.markdown(f"**{row['title']}**")
                with cols[1]:
                    st.markdown(f"⭐ {row['avg_rating']:.1f}")
                with cols[2]:
                    st.markdown(f"📊 {row['num_ratings']}")
    
    with col2:
        st.subheader("⭐ Top Rated Movies")
        top_rated = loader.get_top_rated_movies(n=10, min_ratings=100)
        
        for idx, row in top_rated.iterrows():
            with st.container():
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    st.markdown(f"**{row['title']}**")
                with cols[1]:
                    st.markdown(f"⭐ {row['avg_rating']:.1f}")
                with cols[2]:
                    st.markdown(f"📊 {row['num_ratings']}")

# Continue in next message...
# app.py (continued)

def show_data_explorer(loader):
    """Data explorer page"""
    st.title("📊 Data Explorer")
    st.markdown("### Explore the MovieLens Dataset")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎬 Movies", "👥 Users", "⭐ Ratings", "🏷️ Tags"])
    
    # ==================== Movies Tab ====================
    with tab1:
        st.subheader("🎬 Movies Database")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search = st.text_input("🔍 Search movies by title", "")
        
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                ["Title", "Most Rated", "Highest Rated"]
            )
        
        # Genre filter
        all_genres = set()
        for genres in loader.movies['genres'].str.split('|'):
            if genres != ['(no genres listed)']:
                all_genres.update(genres)
        
        selected_genres = st.multiselect(
            "🎭 Filter by genre",
            sorted(list(all_genres)),
            default=[]
        )
        
        # Apply filters
        movies_display = loader.movies.copy()
        
        if search:
            movies_display = movies_display[
                movies_display['title'].str.contains(search, case=False, na=False)
            ]
        
        if selected_genres:
            movies_display = movies_display[
                movies_display['genres'].apply(
                    lambda x: any(g in x for g in selected_genres)
                )
            ]
        
        # Add statistics
        movie_stats = loader.get_movie_stats()
        movies_display = movies_display.merge(
            movie_stats[['movieId', 'num_ratings', 'avg_rating']], 
            on='movieId', 
            how='left'
        )
        movies_display['num_ratings'].fillna(0, inplace=True)
        movies_display['avg_rating'].fillna(0, inplace=True)
        
        # Sort
        if sort_by == "Most Rated":
            movies_display = movies_display.sort_values('num_ratings', ascending=False)
        elif sort_by == "Highest Rated":
            movies_display = movies_display[movies_display['num_ratings'] >= 10]
            movies_display = movies_display.sort_values('avg_rating', ascending=False)
        else:
            movies_display = movies_display.sort_values('title')
        
        # Display results
        st.markdown(f"**Found {len(movies_display)} movies**")
        
        # Show in expandable cards
        for idx, row in movies_display.head(50).iterrows():
            with st.expander(f"🎬 {row['title']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Genre", row['genres'])
                with col2:
                    st.metric("Ratings", f"{int(row['num_ratings'])}")
                with col3:
                    st.metric("Avg Rating", f"{row['avg_rating']:.2f}" if row['avg_rating'] > 0 else "N/A")
        
        # Movie statistics
        st.markdown("---")
        st.subheader("📈 Movie Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Most Rated Movies**")
            top_rated = movie_stats.nlargest(10, 'num_ratings')[
                ['title', 'num_ratings', 'avg_rating']
            ]
            st.dataframe(top_rated, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Highest Rated Movies** (min 50 ratings)")
            highly_rated = movie_stats[movie_stats['num_ratings'] >= 50].nlargest(
                10, 'avg_rating'
            )[['title', 'num_ratings', 'avg_rating']]
            st.dataframe(highly_rated, use_container_width=True, hide_index=True)
    
    # ==================== Users Tab ====================
    with tab2:
        st.subheader("👥 User Statistics")
        
        user_stats = loader.get_user_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", f"{user_stats.shape[0]:,}")
        
        with col2:
            st.metric("Avg Ratings/User", f"{user_stats['num_ratings'].mean():.1f}")
        
        with col3:
            st.metric("Median Ratings/User", f"{user_stats['num_ratings'].median():.0f}")
        
        with col4:
            st.metric("Max Ratings", f"{user_stats['num_ratings'].max():.0f}")
        
        st.markdown("---")
        
        # User activity distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**User Activity Distribution**")
            fig = px.histogram(
                user_stats,
                x='num_ratings',
                nbins=50,
                labels={'num_ratings': 'Number of Ratings'},
                color_discrete_sequence=['#FF4B4B']
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Average Rating Distribution**")
            fig = px.histogram(
                user_stats,
                x='avg_rating',
                nbins=30,
                labels={'avg_rating': 'Average Rating'},
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top users
        st.markdown("---")
        st.subheader("🏆 Most Active Users")
        
        top_users = user_stats.nlargest(20, 'num_ratings')
        
        fig = px.bar(
            top_users,
            x='userId',
            y='num_ratings',
            labels={'userId': 'User ID', 'num_ratings': 'Number of Ratings'},
            color='avg_rating',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(top_users, use_container_width=True, hide_index=True)
    
    # ==================== Ratings Tab ====================
    with tab3:
        st.subheader("⭐ Ratings Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution over time
            if 'timestamp' in loader.ratings.columns:
                st.markdown("**Ratings Over Time**")
                
                ratings_time = loader.ratings.copy()
                ratings_time['date'] = pd.to_datetime(
                    ratings_time['timestamp'], 
                    unit='s'
                )
                ratings_time['year'] = ratings_time['date'].dt.year
                
                yearly_counts = ratings_time['year'].value_counts().sort_index()
                
                fig = px.line(
                    x=yearly_counts.index,
                    y=yearly_counts.values,
                    labels={'x': 'Year', 'y': 'Number of Ratings'},
                    markers=True
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Rating Value Distribution**")
            rating_dist = loader.ratings['rating'].value_counts().sort_index()
            
            fig = px.pie(
                values=rating_dist.values,
                names=rating_dist.index,
                title="Proportion of each rating"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Genre analysis
        st.subheader("🎭 Average Rating by Genre")
        
        genre_ratings = []
        for _, row in loader.movies.iterrows():
            genres = row['genres'].split('|')
            movie_ratings = loader.ratings[
                loader.ratings['movieId'] == row['movieId']
            ]['rating']
            
            if len(movie_ratings) > 0:
                avg_rating = movie_ratings.mean()
                for genre in genres:
                    if genre != '(no genres listed)':
                        genre_ratings.append({
                            'genre': genre,
                            'rating': avg_rating
                        })
        
        genre_df = pd.DataFrame(genre_ratings)
        genre_avg = genre_df.groupby('genre')['rating'].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=genre_avg.values,
            y=genre_avg.index,
            orientation='h',
            labels={'x': 'Average Rating', 'y': 'Genre'},
            color=genre_avg.values,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== Tags Tab ====================
    with tab4:
        if loader.tags is not None:
            st.subheader("🏷️ User Tags Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Tags", f"{len(loader.tags):,}")
            
            with col2:
                st.metric("Unique Tags", f"{loader.tags['tag'].nunique():,}")
            
            with col3:
                st.metric("Tagged Movies", f"{loader.tags['movieId'].nunique():,}")
            
            st.markdown("---")
            
            # Most common tags
            st.markdown("**Most Common Tags**")
            
            tag_counts = loader.tags['tag'].value_counts().head(30)
            
            fig = px.bar(
                y=tag_counts.index,
                x=tag_counts.values,
                orientation='h',
                labels={'x': 'Count', 'y': 'Tag'},
                color=tag_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tag cloud would go here (you can add word cloud if needed)
            
        else:
            st.info("No tags data available in this dataset")


def show_content_based(loader):
    """Content-based filtering page"""
    st.title("🎯 Content-Based Filtering")
    st.markdown("### Recommendations based on movie features")
    
    st.info("""
    **How it works:** This method recommends movies similar to ones you like based on:
    - 🎭 **Genres**: Movies with similar genre combinations
    - 🏷️ **Tags**: User-generated tags and keywords
    - 📊 **TF-IDF**: Term frequency analysis for better matching
    """)
    
    # Train model
    with st.spinner("🔄 Training content-based model..."):
        if st.session_state.cb_model is None:
            st.session_state.cb_model = train_content_based(loader)
        cb_model = st.session_state.cb_model
    
    st.success("✅ Model ready!")
    
    tab1, tab2, tab3 = st.tabs(["🎬 Similar Movies", "👤 User Recommendations", "🎭 Genre Search"])
    
    # ==================== Similar Movies Tab ====================
    with tab1:
        st.subheader("🎬 Find Similar Movies")
        st.markdown("Select a movie to find similar recommendations")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Movie selection
            movie_title = st.selectbox(
                "Select a movie",
                options=loader.movies['title'].values,
                index=0
            )
        
        with col2:
            n_recommendations = st.slider(
                "Number of recommendations",
                min_value=1,
                max_value=20,
                value=10
            )
        
        if st.button("🔍 Get Similar Movies", use_container_width=True):
            movie_id = loader.movies[
                loader.movies['title'] == movie_title
            ]['movieId'].values[0]
            
            with st.spinner("Finding similar movies..."):
                recommendations = cb_model.get_similar_movies(
                    movie_id, 
                    n=n_recommendations
                )
            
            if recommendations is not None:
                st.markdown(f"### Movies similar to **{movie_title}**")
                st.markdown("---")
                
                for idx, row in recommendations.iterrows():
                    display_movie_card(row, score_label="Similarity")
            else:
                st.warning("No recommendations found")
    
    # ==================== User Recommendations Tab ====================
    with tab2:
        st.subheader("👤 Personalized Recommendations")
        st.markdown("Get recommendations based on user's rating history")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            user_id = st.number_input(
                "Enter User ID",
                min_value=1,
                max_value=int(loader.ratings['userId'].max()),
                value=1
            )
        
        with col2:
            n_recommendations = st.slider(
                "Recommendations",
                min_value=1,
                max_value=20,
                value=10,
                key="cb_user_n"
            )
        
        with col3:
            min_rating = st.slider(
                "Min Rating",
                min_value=1.0,
                max_value=5.0,
                value=4.0,
                step=0.5
            )
        
        if st.button("🎬 Get Recommendations", use_container_width=True):
            # Show user's history
            user_ratings = loader.get_user_rated_movies(user_id)
            
            with st.expander(f"📜 User {user_id}'s Rating History ({len(user_ratings)} movies)", expanded=False):
                st.dataframe(
                    user_ratings[['title', 'genres', 'rating']].head(20), 
                    use_container_width=True,
                    hide_index=True
                )
            
            # Get recommendations
            with st.spinner("Generating recommendations..."):
                recommendations = cb_model.recommend_for_user(
                    user_id, 
                    n=n_recommendations,
                    min_rating=min_rating
                )
            
            if recommendations is not None:
                st.markdown(f"### 🎬 Recommended for User {user_id}")
                st.markdown("---")
                
                for idx, row in recommendations.iterrows():
                    display_movie_card(row)
            else:
                st.warning("No recommendations available for this user")
    
    # ==================== Genre Search Tab ====================
    with tab3:
        st.subheader("🎭 Genre-Based Recommendations")
        
        # Get all genres
        all_genres = set()
        for genres in loader.movies['genres'].str.split('|'):
            if genres != ['(no genres listed)']:
                all_genres.update(genres)
        
        selected_genres = st.multiselect(
            "Select genres",
            sorted(list(all_genres)),
            default=['Action', 'Adventure']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_movies = st.slider(
                "Number of movies",
                min_value=5,
                max_value=50,
                value=10
            )
        
        with col2:
            min_ratings = st.slider(
                "Minimum ratings",
                min_value=10,
                max_value=100,
                value=20
            )
        
        if st.button("🔍 Search", use_container_width=True) and selected_genres:
            with st.spinner("Searching..."):
                recommendations = cb_model.get_genre_based_recommendations(
                    selected_genres,
                    n=n_movies,
                    min_ratings=min_ratings
                )
            
            if recommendations is not None and len(recommendations) > 0:
                st.markdown(f"### Top {len(recommendations)} {', '.join(selected_genres)} Movies")
                st.markdown("---")
                
                for idx, row in recommendations.iterrows():
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"**🎬 {row['title']}**")
                            st.markdown(f"*{row['genres']}*")
                        
                        with col2:
                            st.metric("Avg Rating", f"{row['avg_rating']:.2f}")
                        
                        with col3:
                            st.metric("Ratings", f"{int(row['num_ratings'])}")
                        
                        st.markdown("---")
            else:
                st.warning("No movies found for selected genres")


def show_collaborative(loader):
    """Collaborative filtering page"""
    st.title("👥 Collaborative Filtering")
    st.markdown("### Recommendations based on user behavior patterns")
    
    st.info("""
    **How it works:**
    - 👥 **User-User**: Finds users similar to you and recommends their favorite movies
    - 🎬 **Item-Item**: Finds movies similar to ones you liked based on rating patterns
    """)
    
    # Train model
    with st.spinner("🔄 Training collaborative filtering models..."):
        if st.session_state.cf_model is None:
            st.session_state.cf_model = train_collaborative(loader)
        cf_model = st.session_state.cf_model
    
    st.success("✅ Models ready!")
    
    # Method selection
    method = st.radio(
        "Select Method",
        ["👥 User-User Collaborative Filtering", "🎬 Item-Item Collaborative Filtering"],
        horizontal=True
    )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        user_id = st.number_input(
            "Enter User ID",
            min_value=1,
            max_value=int(loader.ratings['userId'].max()),
            value=1,
            key="cf_user_id"
        )
    
    with col2:
        n_recommendations = st.slider(
            "Recommendations",
            min_value=1,
            max_value=20,
            value=10,
            key="cf_n"
        )
    
    with col3:
        if "User-User" in method:
            n_neighbors = st.slider(
                "Similar Users",
                min_value=5,
                max_value=50,
                value=20
            )
        else:
            n_neighbors = 20
    
    if st.button("🎬 Get Recommendations", use_container_width=True):
        # Show user's history
        user_ratings = loader.get_user_rated_movies(user_id)
        
        with st.expander(f"📜 User {user_id}'s Rating History ({len(user_ratings)} movies)", expanded=False):
            st.dataframe(
                user_ratings[['title', 'genres', 'rating']].head(20), 
                use_container_width=True,
                hide_index=True
            )
        
        # Get recommendations
        with st.spinner("Generating recommendations..."):
            if "User-User" in method:
                recommendations = cf_model.user_based_recommend(
                    user_id, 
                    n=n_recommendations,
                    n_similar_users=n_neighbors
                )
                
                # Show similar users
                neighbors = cf_model.get_user_neighbors(user_id, n=10)
                if neighbors is not None:
                    with st.expander("🔍 Similar Users", expanded=False):
                        st.dataframe(neighbors, use_container_width=True, hide_index=True)
            else:
                recommendations = cf_model.item_based_recommend(
                    user_id, 
                    n=n_recommendations
                )
        
        if recommendations is not None:
            st.markdown(f"### 🎬 {method} Recommendations")
            st.markdown("---")
            
            for idx, row in recommendations.iterrows():
                display_movie_card(row, score_label="Predicted Rating")
        else:
            st.warning("No recommendations available for this user")


def show_matrix_factorization(loader):
    """Matrix factorization page"""
    st.title("🔢 Matrix Factorization (SVD)")
    st.markdown("### Advanced recommendations using Singular Value Decomposition")
    
    st.info("""
    **How it works:**
    - 🧮 Decomposes the user-item rating matrix into latent features
    - 🎯 Learns hidden patterns in user preferences and movie characteristics
    - 📊 Predicts ratings for unseen user-movie pairs
    - 🚀 More accurate than traditional methods for large datasets
    """)
    
    # Training parameters
    with st.expander("⚙️ Model Configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            n_factors = st.slider("Number of latent factors", 10, 200, 50, 10)
            n_epochs = st.slider("Training epochs", 5, 30, 10, 5)
        
        with col2:
            lr = st.slider("Learning rate", 0.001, 0.01, 0.005, 0.001, format="%.3f")
            reg = st.slider("Regularization", 0.01, 0.1, 0.02, 0.01, format="%.2f")
        
        st.markdown(f"""
        **Current Configuration:**
        - Factors: {n_factors} | Epochs: {n_epochs}
        - Learning Rate: {lr} | Regularization: {reg}
        """)
    
    # Train model
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🚀 Train Model", use_container_width=True):
            with st.spinner("Training SVD model... This may take a moment."):
                mf_model = MatrixFactorization(loader.ratings, loader.movies)
                mf_model.train(
                    algorithm='SVD',
                    n_factors=n_factors,
                    n_epochs=n_epochs,
                    lr_all=lr,
                    reg_all=reg,
                    verbose=False
                )
                st.session_state.mf_model = mf_model
            
            # Evaluate
            metrics = mf_model.evaluate()
            
            st.success("✅ Model trained successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE", f"{metrics['RMSE']:.4f}", help="Root Mean Square Error - Lower is better")
            with col2:
                st.metric("MAE", f"{metrics['MAE']:.4f}", help="Mean Absolute Error - Lower is better")
    
    with col2:
        if st.session_state.mf_model is not None:
            if st.button("💾 Save Model", use_container_width=True):
                st.session_state.mf_model.save_model('mf_model.pkl')
                st.success("Model saved successfully!")
    
    # Recommendations
    if st.session_state.mf_model is not None:
        st.markdown("---")
        st.subheader("🎬 Get Recommendations")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_id = st.number_input(
                "Enter User ID",
                min_value=1,
                max_value=int(loader.ratings['userId'].max()),
                value=1,
                key="mf_user_id"
            )
        
        with col2:
            n_recommendations = st.slider(
                "Recommendations",
                min_value=1,
                max_value=20,
                value=10,
                key="mf_n"
            )
        
        if st.button("🎬 Get Recommendations", use_container_width=True, key="mf_recommend"):
            mf_model = st.session_state.mf_model
            
            # Show user's history
            user_ratings = loader.get_user_rated_movies(user_id)
            
            with st.expander(f"📜 User {user_id}'s Rating History ({len(user_ratings)} movies)", expanded=False):
                st.dataframe(
                    user_ratings[['title', 'genres', 'rating']].head(20), 
                    use_container_width=True,
                    hide_index=True
                )
            
            # Get recommendations
            with st.spinner("Generating recommendations..."):
                recommendations = mf_model.recommend_for_user(
                    user_id, 
                    n=n_recommendations
                )
            
            if recommendations is not None:
                st.markdown(f"### 🎬 Recommended Movies (SVD)")
                st.markdown("---")
                
                for idx, row in recommendations.iterrows():
                    display_movie_card(row, score_label="Predicted Rating")
    else:
        st.warning("⚠️ Please train the model first")


def show_comparison(loader):
    """Model comparison page"""
    st.title("⚖️ Model Comparison")
    st.markdown("### Compare recommendations from different algorithms")
    
    st.info("""
    This page generates recommendations using all three algorithms and displays them side-by-side 
    for easy comparison.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_id = st.number_input(
            "Enter User ID",
            min_value=1,
            max_value=int(loader.ratings['userId'].max()),
            value=1,
            key="compare_user_id"
        )
    
    with col2:
        n_recommendations = st.slider(
            "Recommendations per model",
            min_value=5,
            max_value=20,
            value=10,
            key="compare_n"
        )
    
    if st.button("⚖️ Compare All Models", use_container_width=True):
        with st.spinner("🔄 Training models and generating recommendations..."):
            # Train all models
            cb_model = train_content_based(loader)
            cf_model = train_collaborative(loader)
            
            if st.session_state.mf_model is None:
                mf_model = train_matrix_factorization(loader, n_factors=50, n_epochs=10)
                st.session_state.mf_model = mf_model
            else:
                mf_model = st.session_state.mf_model
            
            # Get recommendations from all models
            cb_recs = cb_model.recommend_for_user(user_id, n=n_recommendations)
            cf_user_recs = cf_model.user_based_recommend(user_id, n=n_recommendations)
            cf_item_recs = cf_model.item_based_recommend(user_id, n=n_recommendations)
            mf_recs = mf_model.recommend_for_user(user_id, n=n_recommendations)
        
        # Show user's history
        user_ratings = loader.get_user_rated_movies(user_id)
        
        with st.expander(f"📜 User {user_id}'s Rating History ({len(user_ratings)} movies)", expanded=True):
            # Show top rated movies
            top_rated = user_ratings.nlargest(10, 'rating')[['title', 'genres', 'rating']]
            st.dataframe(top_rated, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("## 📊 Recommendations Comparison")
        
        # Display in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Content-Based")
            if cb_recs is not None:
                for idx, row in cb_recs.head(5).iterrows():
                    st.markdown(f"{idx+1}. **{row['title']}**")
                    st.caption(f"   {row['genres']} | Score: {row.get('score', 0):.2f}")
            else:
                st.warning("No recommendations")
            
            st.markdown("---")
            
            st.markdown("### 👥 User-User CF")
            if cf_user_recs is not None:
                for idx, row in cf_user_recs.head(5).iterrows():
                    st.markdown(f"{idx+1}. **{row['title']}**")
                    st.caption(f"   {row['genres']} | Rating: {row['predicted_rating']:.2f}")
            else:
                st.warning("No recommendations")
        
        with col2:
            st.markdown("### 🎬 Item-Item CF")
            if cf_item_recs is not None:
                for idx, row in cf_item_recs.head(5).iterrows():
                    st.markdown(f"{idx+1}. **{row['title']}**")
                    st.caption(f"   {row['genres']} | Rating: {row['predicted_rating']:.2f}")
            else:
                st.warning("No recommendations")
            
            st.markdown("---")
            
            st.markdown("### 🔢 Matrix Factorization")
            if mf_recs is not None:
                for idx, row in mf_recs.head(5).iterrows():
                    st.markdown(f"{idx+1}. **{row['title']}**")
                    st.caption(f"   {row['genres']} | Rating: {row['predicted_rating']:.2f}")
            else:
                st.warning("No recommendations")
        
        # Analysis
        st.markdown("---")
        st.markdown("## 🔍 Analysis")
        
        # Find common recommendations
        all_recs = []
        if cb_recs is not None:
            all_recs.append(set(cb_recs['movieId'].values))
        if cf_user_recs is not None:
            all_recs.append(set(cf_user_recs['movieId'].values))
        if cf_item_recs is not None:
            all_recs.append(set(cf_item_recs['movieId'].values))
        if mf_recs is not None:
            all_recs.append(set(mf_recs['movieId'].values))
        
        if len(all_recs) >= 2:
            common = set.intersection(*all_recs)
            
            if common:
                st.success(f"✅ **{len(common)} movies recommended by all algorithms:**")
                common_movies = loader.movies[loader.movies['movieId'].isin(common)]
                for _, movie in common_movies.iterrows():
                    st.markdown(f"- **{movie['title']}** ({movie['genres']})")
            else:
                st.info("ℹ️ No movies recommended by all algorithms")
                
                # Check pairwise overlaps
                if len(all_recs) == 4:
                    st.markdown("**Pairwise overlaps:**")
                    labels = ["Content-Based", "User-User CF", "Item-Item CF", "Matrix Factorization"]
                    for i in range(len(all_recs)):
                        for j in range(i+1, len(all_recs)):
                            overlap = len(all_recs[i] & all_recs[j])
                            st.markdown(f"- {labels[i]} ∩ {labels[j]}: {overlap} movies")


def show_evaluation(loader):
    """Evaluation page"""
    st.title("📈 Model Evaluation")
    st.markdown("### Comprehensive evaluation of recommendation quality")
    
    st.info("""
    Evaluate models using multiple metrics:
    - **RMSE/MAE**: Prediction accuracy
    - **Precision@K**: Relevance of top-K recommendations  
    - **Recall@K**: Coverage of relevant items
    - **NDCG@K**: Ranking quality
    """)
    
    evaluator = RecommenderEvaluator(loader.ratings, loader.movies)
    
    tab1, tab2, tab3 = st.tabs(["📊 Dataset Statistics", "🎯 Model Performance", "📉 Visualizations"])
    
    # ==================== Dataset Statistics ====================
    with tab1:
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        info = loader.get_dataset_info()
        
        with col1:
            st.metric("Users", f"{info['num_users']:,}")
            st.metric("Movies", f"{info['num_movies']:,}")
        
        with col2:
            st.metric("Ratings", f"{info['num_ratings']:,}")
            sparsity = info['sparsity']
            st.metric("Sparsity", f"{sparsity*100:.2f}%")
        
        with col3:
            st.metric("Avg Rating", f"{info['avg_rating']:.2f}")
            st.metric("Std Rating", f"{info['std_rating']:.2f}")
        
        with col4:
            st.metric("Ratings/User", f"{info['avg_ratings_per_user']:.1f}")
            st.metric("Ratings/Movie", f"{info['avg_ratings_per_movie']:.1f}")
        
        st.markdown("---")
        
        # Distribution visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Rating Distribution**")
            rating_dist = loader.ratings['rating'].value_counts().sort_index()
            fig = px.bar(
                x=rating_dist.index,
                y=rating_dist.values,
                labels={'x': 'Rating', 'y': 'Count'},
                color=rating_dist.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**User Activity Distribution**")
            user_activity = loader.ratings.groupby('userId').size()
            fig = px.histogram(
                x=user_activity.values,
                nbins=50,
                labels={'x': 'Number of Ratings', 'y': 'Number of Users'},
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== Model Performance ====================
    with tab2:
        st.subheader("🎯 Evaluate Matrix Factorization Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cv_folds = st.slider("Cross-validation folds", 2, 10, 5)
        
        with col2:
            n_factors = st.slider("Number of factors", 20, 100, 50, 10)
        
        if st.button("🚀 Run Evaluation", use_container_width=True):
            with st.spinner("Training and evaluating model... This may take a minute."):
                mf_model = MatrixFactorization(loader.ratings, loader.movies)
                
                # Single train-test evaluation
                mf_model.train(n_factors=n_factors, n_epochs=10, verbose=False)
                metrics = mf_model.evaluate()
                
                st.markdown("### Single Split Evaluation")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                
                with col2:
                    st.metric("MAE", f"{metrics['MAE']:.4f}")
                
                st.markdown("---")
                
                # Cross-validation
                st.markdown(f"### {cv_folds}-Fold Cross-Validation")
                
                cv_results = mf_model.cross_validate_model(
                    cv=cv_folds, 
                    n_factors=n_factors,
                    n_epochs=10,
                    verbose=False
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "RMSE", 
                        f"{cv_results['test_rmse_mean']:.4f}",
                        delta=f"±{cv_results['test_rmse_std']:.4f}"
                    )
                    st.metric(
                        "MAE", 
                        f"{cv_results['test_mae_mean']:.4f}",
                        delta=f"±{cv_results['test_mae_std']:.4f}"
                    )
                
                with col2:
                    st.metric("Fit Time", f"{cv_results['fit_time_mean']:.2f}s")
                    st.metric("Test Time", f"{cv_results['test_time_mean']:.2f}s")
                
                st.success("✅ Evaluation complete!")
    
    # ==================== Visualizations ====================
    with tab3:
        st.subheader("📉 Data Visualizations")
        
        viz_type = st.selectbox(
            "Select visualization",
            [
                "Rating Distribution",
                "User Activity (Top 20)",
                "Movie Popularity (Top 20)",
                "Genre Distribution"
            ]
        )
        
        if viz_type == "Rating Distribution":
            rating_dist = loader.ratings['rating'].value_counts().sort_index()
            fig = px.bar(
                x=rating_dist.index,
                y=rating_dist.values,
                labels={'x': 'Rating', 'y': 'Count'},
                title="Distribution of Ratings",
                color=rating_dist.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "User Activity (Top 20)":
            user_counts = loader.ratings['userId'].value_counts().head(20)
            fig = px.bar(
                x=user_counts.index.astype(str),
                y=user_counts.values,
                labels={'x': 'User ID', 'y': 'Number of Ratings'},
                title="Top 20 Most Active Users",
                color=user_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Movie Popularity (Top 20)":
            movie_counts = loader.ratings['movieId'].value_counts().head(20)
            movie_titles = loader.movies.set_index('movieId')['title'].to_dict()
            labels = [movie_titles.get(mid, f'Movie {mid}')[:40] for mid in movie_counts.index]
            
            fig = go.Figure(go.Bar(
                x=movie_counts.values,
                y=labels,
                orientation='h',
                marker=dict(color=movie_counts.values, colorscale='Reds')
            ))
            fig.update_layout(
                title="Top 20 Most Rated Movies",
                xaxis_title="Number of Ratings",
                yaxis_title="Movie",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Genre Distribution":
            genres = loader.movies['genres'].str.split('|', expand=True).stack()
            genre_counts = genres.value_counts().head(15)
            
            fig = px.pie(
                values=genre_counts.values,
                names=genre_counts.index,
                title="Genre Distribution (Top 15)"
            )
            st.plotly_chart(fig, use_container_width=True)


# Run the app
if __name__ == "__main__":
    main()
