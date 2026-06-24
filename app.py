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

if __name__ == "__main__":
    main()