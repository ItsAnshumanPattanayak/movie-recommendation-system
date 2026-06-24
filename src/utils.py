# src/utils.py

import pandas as pd
import numpy as np

def sample_dataset(ratings_df, movies_df, tags_df=None, 
                   n_users=1000, n_movies=5000, min_ratings_per_user=20):
    """
    Sample a smaller dataset for testing/demo purposes
    
    Args:
        ratings_df: Original ratings DataFrame
        movies_df: Original movies DataFrame
        tags_df: Original tags DataFrame (optional)
        n_users: Number of users to sample
        n_movies: Number of movies to sample
        min_ratings_per_user: Minimum ratings per user
        
    Returns:
        tuple: (sampled_ratings, sampled_movies, sampled_tags)
    """
    print(f"\nSampling dataset...")
    print(f"  Original: {len(ratings_df):,} ratings, {ratings_df['userId'].nunique():,} users")
    
    # Get most active users
    user_counts = ratings_df['userId'].value_counts()
    active_users = user_counts[user_counts >= min_ratings_per_user].head(n_users).index
    
    # Get most rated movies
    movie_counts = ratings_df['movieId'].value_counts().head(n_movies).index
    
    # Filter ratings
    sampled_ratings = ratings_df[
        (ratings_df['userId'].isin(active_users)) &
        (ratings_df['movieId'].isin(movie_counts))
    ].copy()
    
    # Filter movies
    sampled_movies = movies_df[
        movies_df['movieId'].isin(sampled_ratings['movieId'].unique())
    ].copy()
    
    # Filter tags
    sampled_tags = None
    if tags_df is not None:
        sampled_tags = tags_df[
            (tags_df['userId'].isin(sampled_ratings['userId'].unique())) &
            (tags_df['movieId'].isin(sampled_ratings['movieId'].unique()))
        ].copy()
    
    print(f"  Sampled: {len(sampled_ratings):,} ratings, {sampled_ratings['userId'].nunique():,} users, {sampled_ratings['movieId'].nunique():,} movies")
    
    return sampled_ratings, sampled_movies, sampled_tags


def check_dataset_size(ratings_df):
    """
    Check if dataset is too large and recommend sampling
    
    Args:
        ratings_df: Ratings DataFrame
        
    Returns:
        dict: Dataset size information
    """
    n_users = ratings_df['userId'].nunique()
    n_movies = ratings_df['movieId'].nunique()
    n_ratings = len(ratings_df)
    
    # Estimate memory requirements
    matrix_size_gb = (n_users * n_movies * 8) / (1024**3)
    
    info = {
        'n_users': n_users,
        'n_movies': n_movies,
        'n_ratings': n_ratings,
        'estimated_matrix_size_gb': matrix_size_gb,
        'is_large': matrix_size_gb > 4,
        'recommend_sampling': matrix_size_gb > 4
    }
    
    return info


def get_popular_movies(ratings_df, movies_df, n=10, min_ratings=50):
    """
    Get most popular movies
    
    Args:
        ratings_df: Ratings DataFrame
        movies_df: Movies DataFrame
        n: Number of movies to return
        min_ratings: Minimum number of ratings
        
    Returns:
        pd.DataFrame: Popular movies
    """
    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    movie_stats.columns = ['movieId', 'num_ratings', 'avg_rating']
    
    movie_stats = movie_stats[movie_stats['num_ratings'] >= min_ratings]
    movie_stats = movie_stats.nlargest(n, 'num_ratings')
    
    # Merge with movie info
    result = movie_stats.merge(
        movies_df[['movieId', 'title', 'genres']], 
        on='movieId'
    )
    
    return result


def get_top_rated_movies(ratings_df, movies_df, n=10, min_ratings=50):
    """
    Get top rated movies
    
    Args:
        ratings_df: Ratings DataFrame
        movies_df: Movies DataFrame
        n: Number of movies to return
        min_ratings: Minimum number of ratings
        
    Returns:
        pd.DataFrame: Top rated movies
    """
    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    movie_stats.columns = ['movieId', 'num_ratings', 'avg_rating']
    
    movie_stats = movie_stats[movie_stats['num_ratings'] >= min_ratings]
    movie_stats = movie_stats.nlargest(n, 'avg_rating')
    
    # Merge with movie info
    result = movie_stats.merge(
        movies_df[['movieId', 'title', 'genres']], 
        on='movieId'
    )
    
    return result


if __name__ == "__main__":
    print("Utils module - Helper functions for movie recommendation system")