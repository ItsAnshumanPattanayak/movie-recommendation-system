# src/content_based.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')


class ContentBasedRecommender:
    """
    Content-based recommendation system using movie features
    
    Uses:
    - Movie genres
    - User tags
    - TF-IDF vectorization
    - Cosine similarity
    """
    
    def __init__(self, movies_df, ratings_df, tags_df=None):
        """
        Initialize content-based recommender
        
        Args:
            movies_df (pd.DataFrame): Movies data
            ratings_df (pd.DataFrame): Ratings data
            tags_df (pd.DataFrame): Tags data (optional)
        """
        self.movies = movies_df.copy()
        self.ratings = ratings_df.copy()
        self.tags = tags_df.copy() if tags_df is not None else None
        
        self.similarity_matrix = None
        self.movie_features = None
        self.feature_names = None
        self.tfidf = None
        
        print("✓ Content-Based Recommender initialized")
    
    def create_movie_features(self, use_tags=True, max_features=100):
        """
        Create feature vectors for movies
        
        Args:
            use_tags (bool): Include user tags in features
            max_features (int): Maximum number of TF-IDF features
            
        Returns:
            scipy.sparse matrix: Feature matrix
        """
        print("\nCreating movie features...")
        
        # Process genres
        self.movies['genres_processed'] = self.movies['genres'].str.replace('|', ' ')
        self.movies['genres_processed'] = self.movies['genres_processed'].str.replace('-', ' ')
        
        # Process tags if available
        if use_tags and self.tags is not None:
            print("  ✓ Including user tags")
            
            # Aggregate tags for each movie - FIXED: Handle NaN values
            # Remove NaN values and convert to string properly
            movie_tags = self.tags.dropna(subset=['tag']).copy()  # Drop NaN tags
            movie_tags['tag'] = movie_tags['tag'].astype(str).str.lower()  # Convert to string
            
            movie_tags = movie_tags.groupby('movieId')['tag'].apply(
                lambda x: ' '.join(x)  # Now all values are strings
            ).reset_index()
            
            # Merge with movies
            self.movies = self.movies.merge(movie_tags, on='movieId', how='left')
            self.movies['tag'].fillna('', inplace=True)
            
            # Combine genres and tags
            self.movies['features'] = (
                self.movies['genres_processed'] + ' ' + 
                self.movies['tag']
            )
        else:
            self.movies['features'] = self.movies['genres_processed']
        
        # Create TF-IDF vectors
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1
        )
        
        self.movie_features = self.tfidf.fit_transform(self.movies['features'])
        self.feature_names = self.tfidf.get_feature_names_out()
        
        print(f"  ✓ Feature matrix shape: {self.movie_features.shape}")
        print(f"  ✓ Number of features: {len(self.feature_names)}")
        
        return self.movie_features
    
    def compute_similarity(self, metric='cosine'):
        """
        Compute similarity matrix between movies
        
        Args:
            metric (str): Similarity metric ('cosine' or 'linear')
            
        Returns:
            np.ndarray: Similarity matrix
        """
        if self.movie_features is None:
            self.create_movie_features()
        
        print(f"\nComputing {metric} similarity...")
        
        if metric == 'cosine':
            self.similarity_matrix = cosine_similarity(self.movie_features)
        elif metric == 'linear':
            self.similarity_matrix = linear_kernel(self.movie_features)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        print(f"  ✓ Similarity matrix shape: {self.similarity_matrix.shape}")
        
        return self.similarity_matrix
    
    def get_similar_movies(self, movie_id, n=10, return_scores=True):
        """
        Get N most similar movies to the given movie
        
        Args:
            movie_id (int): ID of the movie
            n (int): Number of recommendations
            return_scores (bool): Include similarity scores
            
        Returns:
            pd.DataFrame: Similar movies
        """
        if self.similarity_matrix is None:
            self.compute_similarity()
        
        # Get movie index
        try:
            idx = self.movies[self.movies['movieId'] == movie_id].index[0]
        except IndexError:
            print(f"✗ Movie ID {movie_id} not found")
            return None
        
        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort by similarity (excluding the movie itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        # Create recommendations DataFrame
        recommendations = self.movies.iloc[movie_indices][
            ['movieId', 'title', 'genres']
        ].copy()
        
        if return_scores:
            recommendations['similarity_score'] = similarity_scores
        
        return recommendations.reset_index(drop=True)
    
    def get_similar_movies_by_title(self, title, n=10):
        """
        Get similar movies by title (with partial matching)
        
        Args:
            title (str): Movie title
            n (int): Number of recommendations
            
        Returns:
            pd.DataFrame: Similar movies
        """
        # Find movie
        matches = self.movies[
            self.movies['title'].str.contains(title, case=False, na=False)
        ]
        
        if len(matches) == 0:
            print(f"✗ No movie found matching '{title}'")
            return None
        
        if len(matches) > 1:
            print(f"Found {len(matches)} matches. Using first match:")
            print(f"  → {matches.iloc[0]['title']}")
        
        movie_id = matches.iloc[0]['movieId']
        
        return self.get_similar_movies(movie_id, n=n)
    
    def recommend_for_user(self, user_id, n=10, min_rating=4.0):
        """
        Recommend movies for a user based on their rating history
        
        Args:
            user_id (int): ID of the user
            n (int): Number of recommendations
            min_rating (float): Minimum rating to consider as "liked"
            
        Returns:
            pd.DataFrame: Recommended movies
        """
        # Get user's rated movies
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        
        if len(user_ratings) == 0:
            print(f"✗ No ratings found for user {user_id}")
            return None
        
        # Get highly rated movies
        liked_movies = user_ratings[user_ratings['rating'] >= min_rating]
        
        if len(liked_movies) == 0:
            # If no movies rated >= min_rating, use top rated movies
            liked_movies = user_ratings.nlargest(min(5, len(user_ratings)), 'rating')
        
        # Get similar movies for each liked movie
        all_recommendations = []
        
        for _, row in liked_movies.iterrows():
            similar = self.get_similar_movies(
                row['movieId'], 
                n=20,  # Get more to have variety
                return_scores=True
            )
            
            if similar is not None:
                similar['source_movie_id'] = row['movieId']
                similar['source_rating'] = row['rating']
                all_recommendations.append(similar)
        
        if not all_recommendations:
            print(f"✗ Could not generate recommendations for user {user_id}")
            return None
        
        # Combine all recommendations
        recommendations = pd.concat(all_recommendations, ignore_index=True)
        
        # Remove movies already rated by user
        rated_movie_ids = set(user_ratings['movieId'].values)
        recommendations = recommendations[
            ~recommendations['movieId'].isin(rated_movie_ids)
        ]
        
        # Calculate weighted score
        recommendations['score'] = (
            recommendations['similarity_score'] * 
            recommendations['source_rating']
        )
        
        # Aggregate by movie (handle duplicates)
        recommendations = recommendations.groupby('movieId').agg({
            'title': 'first',
            'genres': 'first',
            'similarity_score': 'mean',
            'score': 'mean'
        }).reset_index()
        
        # Sort by score
        recommendations = recommendations.sort_values('score', ascending=False)
        
        return recommendations[['movieId', 'title', 'genres', 'score']].head(n)
    
    def get_genre_based_recommendations(self, genres, n=10, min_ratings=20):
        """
        Recommend movies based on specific genres
        
        Args:
            genres (list): List of genres
            n (int): Number of recommendations
            min_ratings (int): Minimum number of ratings required
            
        Returns:
            pd.DataFrame: Recommended movies
        """
        # Filter movies by genre
        genre_pattern = '|'.join(genres)
        genre_movies = self.movies[
            self.movies['genres'].str.contains(genre_pattern, case=False, na=False)
        ]
        
        # Get movie ratings
        movie_stats = self.ratings.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        movie_stats.columns = ['movieId', 'num_ratings', 'avg_rating']
        
        # Merge with genre movies
        recommendations = genre_movies.merge(movie_stats, on='movieId', how='left')
        
        # Filter by minimum ratings
        recommendations = recommendations[
            recommendations['num_ratings'] >= min_ratings
        ]
        
        # Sort by average rating
        recommendations = recommendations.sort_values('avg_rating', ascending=False)
        
        return recommendations[
            ['movieId', 'title', 'genres', 'num_ratings', 'avg_rating']
        ].head(n)
    
    def explain_recommendation(self, movie_id, recommended_movie_id):
        """
        Explain why a movie was recommended
        
        Args:
            movie_id (int): Source movie ID
            recommended_movie_id (int): Recommended movie ID
            
        Returns:
            dict: Explanation details
        """
        if self.similarity_matrix is None:
            self.compute_similarity()
        
        try:
            idx1 = self.movies[self.movies['movieId'] == movie_id].index[0]
            idx2 = self.movies[self.movies['movieId'] == recommended_movie_id].index[0]
        except IndexError:
            return None
        
        similarity = self.similarity_matrix[idx1, idx2]
        
        movie1 = self.movies.iloc[idx1]
        movie2 = self.movies.iloc[idx2]
        
        # Get feature vectors
        features1 = self.movie_features[idx1].toarray()[0]
        features2 = self.movie_features[idx2].toarray()[0]
        
        # Find common important features
        common_features = []
        for i, (f1, f2) in enumerate(zip(features1, features2)):
            if f1 > 0 and f2 > 0:
                common_features.append({
                    'feature': self.feature_names[i],
                    'weight': f1 * f2
                })
        
        common_features = sorted(common_features, key=lambda x: x['weight'], reverse=True)
        
        explanation = {
            'similarity_score': similarity,
            'movie1': {
                'id': movie_id,
                'title': movie1['title'],
                'genres': movie1['genres']
            },
            'movie2': {
                'id': recommended_movie_id,
                'title': movie2['title'],
                'genres': movie2['genres']
            },
            'common_features': common_features[:5]  # Top 5 features
        }
        
        return explanation


# Test the content-based recommender
if __name__ == "__main__":
    from data_loader import MovieLensLoader
    
    print("="*60)
    print("Testing Content-Based Recommender")
    print("="*60 + "\n")
    
    # Load data
    loader = MovieLensLoader(data_path='dataset')
    
    if loader.load_data(verbose=False):
        # Sample if large
        if len(loader.ratings) > 1_000_000:
            print("Large dataset detected. Sampling...")
            loader.sample_data(n_users=1000, n_movies=3000)
        
        # Create recommender
        recommender = ContentBasedRecommender(
            loader.movies,
            loader.ratings,
            loader.tags
        )
        
        # Test 1: Similar movies
        print("\n" + "="*60)
        print("Test 1: Find similar movies to 'Toy Story'")
        print("="*60)
        similar = recommender.get_similar_movies_by_title('Toy Story', n=5)
        if similar is not None:
            print(similar.to_string(index=False))
        
        # Test 2: User recommendations
        print("\n" + "="*60)
        print("Test 2: Recommendations for User 1")
        print("="*60)
        user_recs = recommender.recommend_for_user(1, n=5)
        if user_recs is not None:
            print(user_recs.to_string(index=False))
        
        # Test 3: Genre-based recommendations
        print("\n" + "="*60)
        print("Test 3: Top Action & Adventure movies")
        print("="*60)
        genre_recs = recommender.get_genre_based_recommendations(
            ['Action', 'Adventure'], 
            n=5
        )
        print(genre_recs.to_string(index=False))
        
        print("\n✓ All tests completed!")