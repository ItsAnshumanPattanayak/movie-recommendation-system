# src/collaborative_filtering.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.spatial.distance import correlation
import warnings
warnings.filterwarnings('ignore')


class CollaborativeFiltering:
    """
    Collaborative Filtering Recommender System
    
    Implements:
    - User-User Collaborative Filtering
    - Item-Item Collaborative Filtering
    - Multiple similarity metrics
    """
    
    def __init__(self, ratings_df, movies_df):
        """
        Initialize collaborative filtering recommender
        
        Args:
            ratings_df (pd.DataFrame): User ratings
            movies_df (pd.DataFrame): Movie information
        """
        self.ratings = ratings_df.copy()
        self.movies = movies_df.copy()
        
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        
        self.user_mean_ratings = None
        
        print("✓ Collaborative Filtering initialized")
    
    def create_user_item_matrix(self, normalize=False):
        """
        Create user-item rating matrix
        
        Args:
            normalize (bool): Normalize by user mean
            
        Returns:
            pd.DataFrame: User-item matrix
        """
        print("\nCreating user-item matrix...")
        
        # Create pivot table
        self.user_item_matrix = self.ratings.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        
        if normalize:
            # Calculate user mean ratings (only for rated items)
            user_ratings = self.ratings.groupby('userId')['rating'].mean()
            self.user_mean_ratings = user_ratings
            
            # Normalize by subtracting user mean
            for user_id in self.user_item_matrix.index:
                user_mean = user_ratings.get(user_id, 0)
                self.user_item_matrix.loc[user_id] = \
                    self.user_item_matrix.loc[user_id].apply(
                        lambda x: x - user_mean if x > 0 else 0
                    )
        
        print(f"  ✓ Matrix shape: {self.user_item_matrix.shape}")
        print(f"  ✓ Sparsity: {(self.user_item_matrix == 0).sum().sum() / self.user_item_matrix.size:.2%}")
        
        return self.user_item_matrix
    
    def compute_user_similarity(self, metric='cosine', min_support=3):
        """
        Compute user-user similarity
        
        Args:
            metric (str): Similarity metric ('cosine', 'pearson')
            min_support (int): Minimum common ratings required
            
        Returns:
            pd.DataFrame: User similarity matrix
        """
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        print(f"\nComputing user-user similarity ({metric})...")
        
        if metric == 'cosine':
            # Cosine similarity
            similarity = cosine_similarity(self.user_item_matrix)
        elif metric == 'pearson':
            # Pearson correlation
            similarity = np.corrcoef(self.user_item_matrix)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Convert to DataFrame
        self.user_similarity = pd.DataFrame(
            similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        # Set diagonal to 0 (user with themselves)
        np.fill_diagonal(self.user_similarity.values, 0)
        
        print(f"  ✓ Similarity matrix shape: {self.user_similarity.shape}")
        
        return self.user_similarity
    
    def compute_item_similarity(self, metric='cosine'):
        """
        Compute item-item similarity
        
        Args:
            metric (str): Similarity metric ('cosine', 'pearson')
            
        Returns:
            pd.DataFrame: Item similarity matrix
        """
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        print(f"\nComputing item-item similarity ({metric})...")
        
        # Transpose to get item-user matrix
        item_user_matrix = self.user_item_matrix.T
        
        if metric == 'cosine':
            similarity = cosine_similarity(item_user_matrix)
        elif metric == 'pearson':
            similarity = np.corrcoef(item_user_matrix)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Convert to DataFrame
        self.item_similarity = pd.DataFrame(
            similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        # Set diagonal to 0
        np.fill_diagonal(self.item_similarity.values, 0)
        
        print(f"  ✓ Similarity matrix shape: {self.item_similarity.shape}")
        
        return self.item_similarity
    
    def user_based_recommend(self, user_id, n=10, n_similar_users=20):
        """
        User-based collaborative filtering recommendations
        
        Args:
            user_id (int): Target user ID
            n (int): Number of recommendations
            n_similar_users (int): Number of similar users to consider
            
        Returns:
            pd.DataFrame: Recommended movies
        """
        if self.user_similarity is None:
            self.compute_user_similarity()
        
        # Check if user exists
        if user_id not in self.user_similarity.index:
            print(f"✗ User {user_id} not found")
            return None
        
        # Get similar users
        similar_users = self.user_similarity.loc[user_id].nlargest(n_similar_users)
        similar_user_ids = similar_users.index.tolist()
        
        # Get ratings from similar users
        similar_user_ratings = self.ratings[
            self.ratings['userId'].isin(similar_user_ids)
        ].copy()
        
        # Get movies already rated by target user
        user_rated_movies = set(
            self.ratings[self.ratings['userId'] == user_id]['movieId']
        )
        
        # Filter out already rated movies
        candidate_ratings = similar_user_ratings[
            ~similar_user_ratings['movieId'].isin(user_rated_movies)
        ]
        
        if len(candidate_ratings) == 0:
            print(f"✗ No new recommendations available for user {user_id}")
            return None
        
        # Add similarity scores
        candidate_ratings['similarity'] = candidate_ratings['userId'].map(similar_users)
        
        # Calculate weighted rating
        candidate_ratings['weighted_rating'] = (
            candidate_ratings['rating'] * candidate_ratings['similarity']
        )
        
        # Aggregate by movie
        recommendations = candidate_ratings.groupby('movieId').agg({
            'weighted_rating': 'sum',
            'similarity': 'sum',
            'rating': 'mean'
        }).reset_index()
        
        # Calculate predicted rating
        recommendations['predicted_rating'] = (
            recommendations['weighted_rating'] / recommendations['similarity']
        )
        
        # Merge with movie information
        recommendations = recommendations.merge(
            self.movies[['movieId', 'title', 'genres']], 
            on='movieId',
            how='left'
        )
        
        # Sort by predicted rating
        recommendations = recommendations.sort_values(
            'predicted_rating', 
            ascending=False
        )
        
        return recommendations[
            ['movieId', 'title', 'genres', 'predicted_rating']
        ].head(n)
    
    def item_based_recommend(self, user_id, n=10, n_similar_items=20):
        """
        Item-based collaborative filtering recommendations
        
        Args:
            user_id (int): Target user ID
            n (int): Number of recommendations
            n_similar_items (int): Number of similar items per rated item
            
        Returns:
            pd.DataFrame: Recommended movies
        """
        if self.item_similarity is None:
            self.compute_item_similarity()
        
        # Get user's rated movies
        user_ratings = self.ratings[self.ratings['userId'] == user_id][
            ['movieId', 'rating']
        ]
        
        if len(user_ratings) == 0:
            print(f"✗ No ratings found for user {user_id}")
            return None
        
        # Calculate scores for all candidate movies
        movie_scores = {}
        
        for _, row in user_ratings.iterrows():
            movie_id = row['movieId']
            rating = row['rating']
            
            # Check if movie exists in similarity matrix
            if movie_id not in self.item_similarity.index:
                continue
            
            # Get similar movies
            similar_movies = self.item_similarity.loc[movie_id].nlargest(n_similar_items)
            
            # Calculate weighted scores
            for similar_movie_id, similarity in similar_movies.items():
                # Skip if already rated
                if similar_movie_id in user_ratings['movieId'].values:
                    continue
                
                if similar_movie_id not in movie_scores:
                    movie_scores[similar_movie_id] = {
                        'weighted_score': 0,
                        'similarity_sum': 0
                    }
                
                movie_scores[similar_movie_id]['weighted_score'] += similarity * rating
                movie_scores[similar_movie_id]['similarity_sum'] += abs(similarity)
        
        if not movie_scores:
            print(f"✗ No recommendations available for user {user_id}")
            return None
        
        # Calculate predicted ratings
        recommendations = []
        for movie_id, scores in movie_scores.items():
            if scores['similarity_sum'] > 0:
                predicted_rating = scores['weighted_score'] / scores['similarity_sum']
                recommendations.append({
                    'movieId': movie_id,
                    'predicted_rating': predicted_rating
                })
        
        # Convert to DataFrame
        recommendations = pd.DataFrame(recommendations)
        
        # Merge with movie information
        recommendations = recommendations.merge(
            self.movies[['movieId', 'title', 'genres']], 
            on='movieId',
            how='left'
        )
        
        # Sort by predicted rating
        recommendations = recommendations.sort_values(
            'predicted_rating', 
            ascending=False
        )
        
        return recommendations[
            ['movieId', 'title', 'genres', 'predicted_rating']
        ].head(n)
    
    def predict_rating(self, user_id, movie_id, method='item'):
        """
        Predict rating for a user-movie pair
        
        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
            method (str): 'user' or 'item' based
            
        Returns:
            float: Predicted rating
        """
        if method == 'user':
            if self.user_similarity is None:
                self.compute_user_similarity()
            
            # Get similar users who rated this movie
            movie_raters = self.ratings[
                self.ratings['movieId'] == movie_id
            ]['userId'].unique()
            
            if user_id not in self.user_similarity.index:
                return self.ratings['rating'].mean()
            
            similar_users = self.user_similarity.loc[user_id, movie_raters]
            similar_users = similar_users[similar_users > 0].nlargest(20)
            
            if len(similar_users) == 0:
                return self.ratings['rating'].mean()
            
            # Get ratings from similar users
            similar_ratings = self.ratings[
                (self.ratings['userId'].isin(similar_users.index)) &
                (self.ratings['movieId'] == movie_id)
            ]
            
            # Calculate weighted average
            weighted_sum = (similar_ratings['rating'].values * 
                          similar_users.loc[similar_ratings['userId']].values).sum()
            similarity_sum = similar_users.sum()
            
            if similarity_sum > 0:
                return weighted_sum / similarity_sum
            else:
                return self.ratings['rating'].mean()
        
        else:  # item-based
            if self.item_similarity is None:
                self.compute_item_similarity()
            
            # Get user's rated movies
            user_ratings = self.ratings[self.ratings['userId'] == user_id]
            
            if movie_id not in self.item_similarity.index:
                return self.ratings['rating'].mean()
            
            # Get similar movies that user has rated
            rated_movie_ids = user_ratings['movieId'].unique()
            similar_movies = self.item_similarity.loc[movie_id, rated_movie_ids]
            similar_movies = similar_movies[similar_movies > 0].nlargest(20)
            
            if len(similar_movies) == 0:
                return self.ratings['rating'].mean()
            
            # Get user's ratings for similar movies
            similar_ratings = user_ratings[
                user_ratings['movieId'].isin(similar_movies.index)
            ]
            
            # Calculate weighted average
            weighted_sum = (similar_ratings['rating'].values * 
                          similar_movies.loc[similar_ratings['movieId']].values).sum()
            similarity_sum = similar_movies.sum()
            
            if similarity_sum > 0:
                return weighted_sum / similarity_sum
            else:
                return self.ratings['rating'].mean()
    
    def get_user_neighbors(self, user_id, n=10):
        """
        Get most similar users (neighbors)
        
        Args:
            user_id (int): Target user ID
            n (int): Number of neighbors
            
        Returns:
            pd.DataFrame: Similar users with similarity scores
        """
        if self.user_similarity is None:
            self.compute_user_similarity()
        
        if user_id not in self.user_similarity.index:
            print(f"✗ User {user_id} not found")
            return None
        
        neighbors = self.user_similarity.loc[user_id].nlargest(n)
        
        # Get user statistics
        user_stats = self.ratings.groupby('userId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        user_stats.columns = ['userId', 'num_ratings', 'avg_rating']
        
        neighbors_df = pd.DataFrame({
            'userId': neighbors.index,
            'similarity': neighbors.values
        })
        
        neighbors_df = neighbors_df.merge(user_stats, on='userId', how='left')
        
        return neighbors_df


# Test the collaborative filtering
if __name__ == "__main__":
    from data_loader import MovieLensLoader
    
    print("="*60)
    print("Testing Collaborative Filtering")
    print("="*60 + "\n")
    
    # Load data
    loader = MovieLensLoader(data_path='dataset')
    
    if loader.load_data(verbose=False):
        # Create recommender
        cf = CollaborativeFiltering(loader.ratings, loader.movies)
        
        # Test 1: User-based recommendations
        print("\n" + "="*60)
        print("Test 1: User-based recommendations for User 1")
        print("="*60)
        user_recs = cf.user_based_recommend(1, n=5)
        if user_recs is not None:
            print(user_recs.to_string(index=False))
        
        # Test 2: Item-based recommendations
        print("\n" + "="*60)
        print("Test 2: Item-based recommendations for User 1")
        print("="*60)
        item_recs = cf.item_based_recommend(1, n=5)
        if item_recs is not None:
            print(item_recs.to_string(index=False))
        
        # Test 3: Get similar users
        print("\n" + "="*60)
        print("Test 3: Find similar users to User 1")
        print("="*60)
        neighbors = cf.get_user_neighbors(1, n=5)
        if neighbors is not None:
            print(neighbors.to_string(index=False))
        
        print("\n✓ All tests completed!")