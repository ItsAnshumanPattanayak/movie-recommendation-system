# src/matrix_factorization.py

import pandas as pd
import numpy as np
from surprise import SVD, NMF, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings('ignore')


class MatrixFactorization:
    """
    Matrix Factorization using SVD (Singular Value Decomposition)
    
    Uses the Surprise library for efficient implementation
    Implements SVD and NMF algorithms
    """
    
    def __init__(self, ratings_df, movies_df):
        """
        Initialize matrix factorization recommender
        
        Args:
            ratings_df (pd.DataFrame): User ratings
            movies_df (pd.DataFrame): Movie information
        """
        self.ratings = ratings_df.copy()
        self.movies = movies_df.copy()
        
        self.model = None
        self.trainset = None
        self.testset = None
        self.full_trainset = None
        
        self.algorithm = 'SVD'
        
        print("✓ Matrix Factorization initialized")
    
    def prepare_data(self, test_size=0.2, rating_scale=(0.5, 5.0)):
        """
        Prepare data for Surprise library
        
        Args:
            test_size (float): Proportion of test set
            rating_scale (tuple): Min and max rating values
            
        Returns:
            tuple: (trainset, testset)
        """
        print("\nPreparing data for training...")
        
        # Define rating scale
        reader = Reader(rating_scale=rating_scale)
        
        # Load data into Surprise format
        data = Dataset.load_from_df(
            self.ratings[['userId', 'movieId', 'rating']], 
            reader
        )
        
        # Split into train and test
        self.trainset, self.testset = train_test_split(
            data, 
            test_size=test_size,
            random_state=42
        )
        
        # Build full trainset for final predictions
        self.full_trainset = data.build_full_trainset()
        
        print(f"  ✓ Training set: {self.trainset.n_ratings:,} ratings")
        print(f"  ✓ Test set: {len(self.testset):,} ratings")
        print(f"  ✓ Users: {self.trainset.n_users:,}")
        print(f"  ✓ Items: {self.trainset.n_items:,}")
        
        return self.trainset, self.testset
    
    def train(self, algorithm='SVD', n_factors=100, n_epochs=20, 
              lr_all=0.005, reg_all=0.02, verbose=True):
        """
        Train matrix factorization model
        
        Args:
            algorithm (str): 'SVD' or 'NMF'
            n_factors (int): Number of latent factors
            n_epochs (int): Number of training iterations
            lr_all (float): Learning rate
            reg_all (float): Regularization parameter
            verbose (bool): Print training progress
            
        Returns:
            Trained model
        """
        if self.trainset is None:
            self.prepare_data()
        
        print(f"\nTraining {algorithm} model...")
        print(f"  Parameters:")
        print(f"    - Factors: {n_factors}")
        print(f"    - Epochs: {n_epochs}")
        print(f"    - Learning rate: {lr_all}")
        print(f"    - Regularization: {reg_all}")
        
        self.algorithm = algorithm
        
        if algorithm == 'SVD':
            self.model = SVD(
                n_factors=n_factors,
                n_epochs=n_epochs,
                lr_all=lr_all,
                reg_all=reg_all,
                random_state=42,
                verbose=verbose
            )
        elif algorithm == 'NMF':
            self.model = NMF(
                n_factors=n_factors,
                n_epochs=n_epochs,
                random_state=42,
                verbose=verbose
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Train on full trainset for better predictions
        self.model.fit(self.full_trainset)
        
        print(f"  ✓ {algorithm} model trained successfully!")
        
        return self.model
    
    def predict_rating(self, user_id, movie_id):
        """
        Predict rating for a user-movie pair
        
        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
            
        Returns:
            float: Predicted rating
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        prediction = self.model.predict(user_id, movie_id)
        
        return prediction.est
    
    def recommend_for_user(self, user_id, n=10, exclude_rated=True):
        """
        Recommend top N movies for a user
        
        Args:
            user_id (int): User ID
            n (int): Number of recommendations
            exclude_rated (bool): Exclude already rated movies
            
        Returns:
            pd.DataFrame: Recommended movies
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get all movie IDs
        all_movie_ids = self.movies['movieId'].values
        
        # Get movies already rated by user
        if exclude_rated:
            rated_movies = set(
                self.ratings[self.ratings['userId'] == user_id]['movieId']
            )
            candidate_movies = [m for m in all_movie_ids if m not in rated_movies]
        else:
            candidate_movies = all_movie_ids
        
        # Predict ratings for all candidate movies
        predictions = []
        for movie_id in candidate_movies:
            pred_rating = self.predict_rating(user_id, movie_id)
            predictions.append({
                'movieId': movie_id,
                'predicted_rating': pred_rating
            })
        
        # Convert to DataFrame
        recommendations = pd.DataFrame(predictions)
        
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
    
    def get_top_n_for_all_users(self, n=10):
        """
        Get top N recommendations for all users
        
        Args:
            n (int): Number of recommendations per user
            
        Returns:
            dict: {user_id: DataFrame of recommendations}
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print(f"\nGenerating top {n} recommendations for all users...")
        
        all_users = self.ratings['userId'].unique()
        recommendations = {}
        
        for i, user_id in enumerate(all_users):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(all_users)} users")
            
            user_recs = self.recommend_for_user(user_id, n=n)
            recommendations[user_id] = user_recs
        
        print(f"  ✓ Generated recommendations for {len(all_users)} users")
        
        return recommendations
    
    def evaluate(self, testset=None, metrics=['RMSE', 'MAE']):
        """
        Evaluate model on test set
        
        Args:
            testset: Test set (uses self.testset if None)
            metrics (list): Metrics to compute
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if testset is None:
            testset = self.testset
        
        print("\nEvaluating model...")
        
        # Make predictions
        predictions = self.model.test(testset)
        
        # Calculate metrics
        from surprise import accuracy
        
        results = {}
        
        if 'RMSE' in metrics:
            rmse = accuracy.rmse(predictions, verbose=False)
            results['RMSE'] = rmse
            print(f"  ✓ RMSE: {rmse:.4f}")
        
        if 'MAE' in metrics:
            mae = accuracy.mae(predictions, verbose=False)
            results['MAE'] = mae
            print(f"  ✓ MAE: {mae:.4f}")
        
        return results
    
    def cross_validate_model(self, cv=5, algorithm='SVD', n_factors=100, 
                            n_epochs=20, verbose=True):
        """
        Perform cross-validation
        
        Args:
            cv (int): Number of folds
            algorithm (str): Algorithm to use
            n_factors (int): Number of factors
            n_epochs (int): Number of epochs
            verbose (bool): Print results
            
        Returns:
            dict: Cross-validation results
        """
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        # Prepare data
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            self.ratings[['userId', 'movieId', 'rating']], 
            reader
        )
        
        # Select algorithm
        if algorithm == 'SVD':
            algo = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=42)
        elif algorithm == 'NMF':
            algo = NMF(n_factors=n_factors, n_epochs=n_epochs, random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Perform cross-validation
        cv_results = cross_validate(
            algo,
            data,
            measures=['RMSE', 'MAE'],
            cv=cv,
            verbose=verbose
        )
        
        # Calculate mean and std
        results = {
            'test_rmse_mean': cv_results['test_rmse'].mean(),
            'test_rmse_std': cv_results['test_rmse'].std(),
            'test_mae_mean': cv_results['test_mae'].mean(),
            'test_mae_std': cv_results['test_mae'].std(),
            'fit_time_mean': cv_results['fit_time'].mean(),
            'test_time_mean': cv_results['test_time'].mean()
        }
        
        print(f"\n  Cross-validation results ({cv} folds):")
        print(f"    RMSE: {results['test_rmse_mean']:.4f} (+/- {results['test_rmse_std']:.4f})")
        print(f"    MAE:  {results['test_mae_mean']:.4f} (+/- {results['test_mae_std']:.4f})")
        print(f"    Fit time: {results['fit_time_mean']:.2f}s")
        print(f"    Test time: {results['test_time_mean']:.2f}s")
        
        return results
    
    def grid_search(self, param_grid=None, cv=3):
        """
        Perform grid search for hyperparameter tuning
        
        Args:
            param_grid (dict): Parameter grid
            cv (int): Number of folds
            
        Returns:
            dict: Best parameters and scores
        """
        if param_grid is None:
            param_grid = {
                'n_factors': [50, 100, 150],
                'n_epochs': [10, 20, 30],
                'lr_all': [0.002, 0.005, 0.01],
                'reg_all': [0.02, 0.05, 0.1]
            }
        
        print("\nPerforming grid search...")
        print(f"  Parameter grid: {param_grid}")
        
        # Prepare data
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            self.ratings[['userId', 'movieId', 'rating']], 
            reader
        )
        
        # Grid search
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=cv, n_jobs=-1)
        gs.fit(data)
        
        print(f"\n  ✓ Best RMSE: {gs.best_score['rmse']:.4f}")
        print(f"  ✓ Best parameters:")
        for param, value in gs.best_params['rmse'].items():
            print(f"      {param}: {value}")
        
        return {
            'best_params': gs.best_params['rmse'],
            'best_rmse': gs.best_score['rmse'],
            'best_mae': gs.best_score['mae']
        }
    
    def get_latent_factors(self, user_id=None, movie_id=None):
        """
        Get latent factor vectors for users or movies
        
        Args:
            user_id (int): User ID (optional)
            movie_id (int): Movie ID (optional)
            
        Returns:
            np.array: Latent factor vector
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if user_id is not None:
            # Get user factors
            try:
                inner_id = self.full_trainset.to_inner_uid(user_id)
                return self.model.pu[inner_id]
            except ValueError:
                print(f"✗ User {user_id} not found in training set")
                return None
        
        if movie_id is not None:
            # Get item factors
            try:
                inner_id = self.full_trainset.to_inner_iid(movie_id)
                return self.model.qi[inner_id]
            except ValueError:
                print(f"✗ Movie {movie_id} not found in training set")
                return None
        
        return None
    
    def save_model(self, filepath='svd_model.pkl'):
        """
        Save trained model to file
        
        Args:
            filepath (str): Path to save file
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'algorithm': self.algorithm,
                'full_trainset': self.full_trainset
            }, f)
        
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath='svd_model.pkl'):
        """
        Load trained model from file
        
        Args:
            filepath (str): Path to model file
            
        Returns:
            bool: True if successful
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.algorithm = data['algorithm']
            self.full_trainset = data['full_trainset']
            
            print(f"✓ Model loaded from {filepath}")
            return True
        
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False


# Test matrix factorization
if __name__ == "__main__":
    from data_loader import MovieLensLoader
    
    print("="*60)
    print("Testing Matrix Factorization")
    print("="*60 + "\n")
    
    # Load data
    loader = MovieLensLoader(data_path='dataset')
    
    if loader.load_data(verbose=False):
        # Create recommender
        mf = MatrixFactorization(loader.ratings, loader.movies)
        
        # Test 1: Train SVD model
        print("\n" + "="*60)
        print("Test 1: Train SVD model")
        print("="*60)
        mf.train(algorithm='SVD', n_factors=50, n_epochs=10, verbose=True)
        
        # Test 2: Evaluate
        print("\n" + "="*60)
        print("Test 2: Evaluate model")
        print("="*60)
        metrics = mf.evaluate()
        
        # Test 3: Get recommendations
        print("\n" + "="*60)
        print("Test 3: Recommendations for User 1")
        print("="*60)
        recs = mf.recommend_for_user(1, n=5)
        if recs is not None:
            print(recs.to_string(index=False))
        
        # Test 4: Cross-validation
        print("\n" + "="*60)
        print("Test 4: Cross-validation (3 folds)")
        print("="*60)
        cv_results = mf.cross_validate_model(cv=3, n_factors=50, n_epochs=10)
        
        print("\n✓ All tests completed!")