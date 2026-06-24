# src/data_loader.py

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import os
from datetime import datetime

class MovieLensLoader:
    """
    Load and preprocess MovieLens dataset
    
    Expected folder structure:
    dataset/
        ├── links.csv
        ├── movies.csv
        ├── ratings.csv
        └── tags.csv
    """
    
    def __init__(self, data_path='dataset'):
        """
        Initialize the data loader
        
        Args:
            data_path (str): Path to the dataset folder containing CSV files
        """
        self.data_path = Path(data_path)
        self.ratings = None
        self.movies = None
        self.tags = None
        self.links = None
        self.user_item_matrix = None
        
        # Verify dataset path exists
        if not self.data_path.exists():
            raise ValueError(f"Dataset path '{data_path}' does not exist!")
    
    def load_data(self, verbose=True):
        """
        Load all MovieLens data files
        
        Args:
            verbose (bool): Print loading information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load ratings.csv
            ratings_file = self.data_path / 'ratings.csv'
            if ratings_file.exists():
                self.ratings = pd.read_csv(ratings_file)
                if verbose:
                    print(f"✓ Loaded {len(self.ratings):,} ratings")
            else:
                raise FileNotFoundError(f"ratings.csv not found in {self.data_path}")
            
            # Load movies.csv
            movies_file = self.data_path / 'movies.csv'
            if movies_file.exists():
                self.movies = pd.read_csv(movies_file)
                if verbose:
                    print(f"✓ Loaded {len(self.movies):,} movies")
            else:
                raise FileNotFoundError(f"movies.csv not found in {self.data_path}")
            
            # Load tags.csv (optional)
            tags_file = self.data_path / 'tags.csv'
            if tags_file.exists():
                self.tags = pd.read_csv(tags_file)
                if verbose:
                    print(f"✓ Loaded {len(self.tags):,} tags")
            else:
                if verbose:
                    print("⚠ tags.csv not found (optional)")
            
            # Load links.csv (optional)
            links_file = self.data_path / 'links.csv'
            if links_file.exists():
                self.links = pd.read_csv(links_file)
                if verbose:
                    print(f"✓ Loaded {len(self.links):,} links")
            else:
                if verbose:
                    print("⚠ links.csv not found (optional)")
            
            if verbose:
                print("\n" + "="*50)
                print("Dataset Summary:")
                print(f"  Users: {self.ratings['userId'].nunique():,}")
                print(f"  Movies: {len(self.movies):,}")
                print(f"  Ratings: {len(self.ratings):,}")
                print(f"  Sparsity: {self._calculate_sparsity():.2%}")
                print("="*50 + "\n")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def _calculate_sparsity(self):
        """Calculate sparsity of the user-item matrix"""
        if self.ratings is None or self.movies is None:
            return 0.0
        
        n_users = self.ratings['userId'].nunique()
        n_movies = len(self.movies)
        n_ratings = len(self.ratings)
        
        sparsity = 1 - (n_ratings / (n_users * n_movies))
        return sparsity
    
    def create_user_item_matrix(self, fill_value=0):
        """
        Create user-item rating matrix
        
        Args:
            fill_value: Value to fill for missing ratings (default: 0)
            
        Returns:
            pd.DataFrame: User-item matrix
        """
        if self.ratings is None:
            raise ValueError("Ratings not loaded. Call load_data() first.")
        
        self.user_item_matrix = self.ratings.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=fill_value
        )
        
        print(f"User-Item Matrix Shape: {self.user_item_matrix.shape}")
        print(f"Memory Usage: {self.user_item_matrix.memory_usage().sum() / 1024**2:.2f} MB")
        
        return self.user_item_matrix
    
    def get_movie_genres(self):
        """
        Process movie genres into a more usable format
        
        Returns:
            tuple: (movies_df with genre columns, list of all genres)
        """
        if self.movies is None:
            raise ValueError("Movies not loaded. Call load_data() first.")
        
        movies_df = self.movies.copy()
        
        # Split genres into list
        movies_df['genres_list'] = movies_df['genres'].str.split('|')
        
        # Get all unique genres
        all_genres = set()
        for genres in movies_df['genres_list']:
            if genres != ['(no genres listed)']:
                all_genres.update(genres)
        
        all_genres = sorted(list(all_genres))
        
        # One-hot encode genres
        for genre in all_genres:
            movies_df[f'genre_{genre}'] = movies_df['genres_list'].apply(
                lambda x: 1 if genre in x else 0
            )
        
        return movies_df, all_genres
    
    def get_user_stats(self):
        """
        Get statistics about users
        
        Returns:
            pd.DataFrame: User statistics
        """
        if self.ratings is None:
            return None
        
        user_stats = self.ratings.groupby('userId').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max'],
            'movieId': 'nunique'
        }).reset_index()
        
        user_stats.columns = ['userId', 'num_ratings', 'avg_rating', 
                              'std_rating', 'min_rating', 'max_rating', 'num_movies']
        
        # Fill NaN std with 0 (users with only one rating)
        user_stats['std_rating'].fillna(0, inplace=True)
        
        return user_stats
    
    def get_movie_stats(self):
        """
        Get statistics about movies
        
        Returns:
            pd.DataFrame: Movie statistics
        """
        if self.ratings is None or self.movies is None:
            return None
        
        movie_stats = self.ratings.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max'],
            'userId': 'nunique'
        }).reset_index()
        
        movie_stats.columns = ['movieId', 'num_ratings', 'avg_rating', 
                               'std_rating', 'min_rating', 'max_rating', 'num_users']
        
        # Fill NaN std with 0
        movie_stats['std_rating'].fillna(0, inplace=True)
        
        # Merge with movie information
        movie_stats = movie_stats.merge(
            self.movies[['movieId', 'title', 'genres']], 
            on='movieId', 
            how='left'
        )
        
        return movie_stats
    
    def get_movie_title(self, movie_id):
        """
        Get movie title by ID
        
        Args:
            movie_id: Movie ID
            
        Returns:
            str: Movie title or None if not found
        """
        if self.movies is None:
            return None
        
        movie = self.movies[self.movies['movieId'] == movie_id]
        if len(movie) > 0:
            return movie.iloc[0]['title']
        return None
    
    def get_movie_id(self, title, partial_match=True):
        """
        Get movie ID by title
        
        Args:
            title (str): Movie title
            partial_match (bool): Allow partial matching
            
        Returns:
            pd.DataFrame: Matching movies
        """
        if self.movies is None:
            return None
        
        if partial_match:
            matches = self.movies[
                self.movies['title'].str.contains(title, case=False, na=False)
            ]
        else:
            matches = self.movies[
                self.movies['title'].str.lower() == title.lower()
            ]
        
        return matches[['movieId', 'title', 'genres']]
    
    def get_user_rated_movies(self, user_id):
        """
        Get all movies rated by a user
        
        Args:
            user_id: User ID
            
        Returns:
            pd.DataFrame: User's rated movies with details
        """
        if self.ratings is None or self.movies is None:
            return None
        
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        
        # Merge with movie information
        user_movies = user_ratings.merge(
            self.movies[['movieId', 'title', 'genres']], 
            on='movieId',
            how='left'
        )
        
        # Sort by rating (descending) and timestamp if available
        if 'timestamp' in user_movies.columns:
            user_movies = user_movies.sort_values(['rating', 'timestamp'], 
                                                   ascending=[False, False])
        else:
            user_movies = user_movies.sort_values('rating', ascending=False)
        
        return user_movies
    
    def get_popular_movies(self, n=10, min_ratings=50):
        """
        Get most popular movies
        
        Args:
            n (int): Number of movies to return
            min_ratings (int): Minimum number of ratings required
            
        Returns:
            pd.DataFrame: Popular movies
        """
        movie_stats = self.get_movie_stats()
        
        popular = movie_stats[movie_stats['num_ratings'] >= min_ratings]
        popular = popular.nlargest(n, 'num_ratings')
        
        return popular[['movieId', 'title', 'genres', 'num_ratings', 'avg_rating']]
    
    def get_top_rated_movies(self, n=10, min_ratings=50):
        """
        Get top rated movies
        
        Args:
            n (int): Number of movies to return
            min_ratings (int): Minimum number of ratings required
            
        Returns:
            pd.DataFrame: Top rated movies
        """
        movie_stats = self.get_movie_stats()
        
        top_rated = movie_stats[movie_stats['num_ratings'] >= min_ratings]
        top_rated = top_rated.nlargest(n, 'avg_rating')
        
        return top_rated[['movieId', 'title', 'genres', 'num_ratings', 'avg_rating']]
    
    def sample_data(self, n_users=1000, n_movies=5000, min_ratings_per_user=20):
        """
        Sample the dataset to a smaller size for testing
        
        Args:
            n_users: Number of users to keep
            n_movies: Number of movies to keep
            min_ratings_per_user: Minimum ratings per user
            
        Returns:
            bool: True if successful
        """
        if self.ratings is None or self.movies is None:
            print("✗ No data loaded to sample from")
            return False
        
        print(f"\n{'='*60}")
        print("SAMPLING DATASET")
        print(f"{'='*60}")
        print(f"Original size: {len(self.ratings):,} ratings")
        
        # Get most active users
        user_counts = self.ratings['userId'].value_counts()
        active_users = user_counts[user_counts >= min_ratings_per_user].head(n_users).index
        
        # Get most rated movies
        movie_counts = self.ratings['movieId'].value_counts().head(n_movies).index
        
        # Filter ratings
        self.ratings = self.ratings[
            (self.ratings['userId'].isin(active_users)) &
            (self.ratings['movieId'].isin(movie_counts))
        ].copy()
        
        # Filter movies
        self.movies = self.movies[
            self.movies['movieId'].isin(self.ratings['movieId'].unique())
        ].copy()
        
        # Filter tags if available
        if self.tags is not None:
            self.tags = self.tags[
                (self.tags['userId'].isin(self.ratings['userId'].unique())) &
                (self.tags['movieId'].isin(self.ratings['movieId'].unique()))
            ].copy()
        
        print(f"Sampled size: {len(self.ratings):,} ratings")
        print(f"Users: {self.ratings['userId'].nunique():,}")
        print(f"Movies: {len(self.movies):,}")
        print(f"{'='*60}\n")
        
        return True

    def save_processed_data(self, filepath='processed_data.pkl'):
        """
        Save processed data to pickle file
        
        Args:
            filepath (str): Path to save file
        """
        data = {
            'ratings': self.ratings,
            'movies': self.movies,
            'tags': self.tags,
            'links': self.links,
            'user_item_matrix': self.user_item_matrix,
            'timestamp': datetime.now()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Data saved to {filepath}")
    
    def load_processed_data(self, filepath='processed_data.pkl'):
        """
        Load processed data from pickle file
        
        Args:
            filepath (str): Path to pickle file
            
        Returns:
            bool: True if successful
        """
        if not os.path.exists(filepath):
            print(f"✗ File {filepath} not found")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.ratings = data['ratings']
            self.movies = data['movies']
            self.tags = data.get('tags')
            self.links = data.get('links')
            self.user_item_matrix = data.get('user_item_matrix')
            
            print(f"✓ Processed data loaded from {filepath}")
            print(f"  Saved on: {data.get('timestamp', 'Unknown')}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading processed data: {e}")
            return False
    
    def get_dataset_info(self):
        """
        Get comprehensive dataset information
        
        Returns:
            dict: Dataset information
        """
        if self.ratings is None or self.movies is None:
            return {}
        
        info = {
            'num_users': self.ratings['userId'].nunique(),
            'num_movies': len(self.movies),
            'num_ratings': len(self.ratings),
            'sparsity': self._calculate_sparsity(),
            'rating_scale': (self.ratings['rating'].min(), self.ratings['rating'].max()),
            'avg_rating': self.ratings['rating'].mean(),
            'std_rating': self.ratings['rating'].std(),
            'avg_ratings_per_user': len(self.ratings) / self.ratings['userId'].nunique(),
            'avg_ratings_per_movie': len(self.ratings) / len(self.movies),
        }
        
        if self.tags is not None:
            info['num_tags'] = len(self.tags)
        
        return info


# Test the data loader
if __name__ == "__main__":
    print("="*60)
    print("Testing MovieLens Data Loader")
    print("="*60 + "\n")
    
    # Initialize loader
    loader = MovieLensLoader(data_path='dataset')
    
    # Load data
    if loader.load_data():
        # Display dataset info
        info = loader.get_dataset_info()
        print("\nDetailed Dataset Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test various methods
        print("\n" + "="*60)
        print("Testing Methods:")
        print("="*60)
        
        # Get popular movies
        print("\n1. Top 5 Popular Movies:")
        print(loader.get_popular_movies(n=5))
        
        # Get top rated movies
        print("\n2. Top 5 Rated Movies:")
        print(loader.get_top_rated_movies(n=5))
        
        # Search movie
        print("\n3. Search for 'Toy Story':")
        print(loader.get_movie_id('Toy Story'))
        
        # Get user's rated movies
        print("\n4. User 1's top rated movies:")
        user_movies = loader.get_user_rated_movies(1)
        if user_movies is not None:
            print(user_movies.head())
        
        # Create user-item matrix
        print("\n5. Creating user-item matrix...")
        loader.create_user_item_matrix()
        
        print("\n✓ All tests completed successfully!")
    else:
        print("\n✗ Failed to load data")