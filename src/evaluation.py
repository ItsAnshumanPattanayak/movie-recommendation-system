# src/evaluation.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class RecommenderEvaluator:
    """
    Comprehensive evaluation for recommendation systems
    
    Metrics:
    - RMSE, MAE (prediction accuracy)
    - Precision@K, Recall@K (ranking quality)
    - Coverage, Diversity (recommendation quality)
    """
    
    def __init__(self, ratings_df, movies_df=None):
        """
        Initialize evaluator
        
        Args:
            ratings_df (pd.DataFrame): User ratings
            movies_df (pd.DataFrame): Movie information (optional)
        """
        self.ratings = ratings_df.copy()
        self.movies = movies_df
        
        print("✓ Recommender Evaluator initialized")
    
    def train_test_split(self, test_size=0.2, random_state=42, 
                         strategy='random'):
        """
        Split data into train and test sets
        
        Args:
            test_size (float): Proportion of test data
            random_state (int): Random seed
            strategy (str): 'random' or 'temporal'
            
        Returns:
            tuple: (train_df, test_df)
        """
        if strategy == 'random':
            from sklearn.model_selection import train_test_split
            
            train, test = train_test_split(
                self.ratings,
                test_size=test_size,
                random_state=random_state
            )
        
        elif strategy == 'temporal':
            # Split based on timestamp if available
            if 'timestamp' not in self.ratings.columns:
                raise ValueError("Temporal split requires 'timestamp' column")
            
            # Sort by timestamp
            sorted_ratings = self.ratings.sort_values('timestamp')
            
            # Split
            split_idx = int(len(sorted_ratings) * (1 - test_size))
            train = sorted_ratings.iloc[:split_idx]
            test = sorted_ratings.iloc[split_idx:]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        print(f"Train set: {len(train):,} ratings")
        print(f"Test set: {len(test):,} ratings")
        
        return train, test
    
    # ==================== Prediction Accuracy Metrics ====================
    
    def calculate_rmse(self, y_true, y_pred):
        """Calculate Root Mean Square Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def calculate_mae(self, y_true, y_pred):
        """Calculate Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    def calculate_mse(self, y_true, y_pred):
        """Calculate Mean Square Error"""
        return mean_squared_error(y_true, y_pred)
    
    # ==================== Ranking Quality Metrics ====================
    
    def precision_at_k(self, recommended_items, relevant_items, k=10):
        """
        Calculate Precision@K
        
        Precision@K = (# of recommended items @K that are relevant) / K
        
        Args:
            recommended_items (list): Ordered list of recommended item IDs
            relevant_items (list): List of relevant item IDs
            k (int): Number of top recommendations to consider
            
        Returns:
            float: Precision@K score
        """
        if k == 0 or len(recommended_items) == 0:
            return 0.0
        
        recommended_at_k = set(recommended_items[:k])
        relevant = set(relevant_items)
        
        return len(recommended_at_k & relevant) / k
    
    def recall_at_k(self, recommended_items, relevant_items, k=10):
        """
        Calculate Recall@K
        
        Recall@K = (# of recommended items @K that are relevant) / (# of relevant items)
        
        Args:
            recommended_items (list): Ordered list of recommended item IDs
            relevant_items (list): List of relevant item IDs
            k (int): Number of top recommendations to consider
            
        Returns:
            float: Recall@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        recommended_at_k = set(recommended_items[:k])
        relevant = set(relevant_items)
        
        return len(recommended_at_k & relevant) / len(relevant)
    
    def f1_at_k(self, recommended_items, relevant_items, k=10):
        """
        Calculate F1@K score
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        """
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        recall = self.recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def average_precision_at_k(self, recommended_items, relevant_items, k=10):
        """
        Calculate Average Precision@K
        
        AP@K = (1/min(m,k)) * Σ(P(i) * rel(i))
        where m is the number of relevant items
        """
        if len(relevant_items) == 0:
            return 0.0
        
        relevant = set(relevant_items)
        score = 0.0
        num_hits = 0.0
        
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return score / min(len(relevant), k)
    
    def mean_average_precision_at_k(self, all_recommendations, all_relevant, k=10):
        """
        Calculate Mean Average Precision@K across all users
        
        MAP@K = (1/|U|) * Σ(AP@K for each user)
        """
        ap_scores = []
        
        for user_id, recommended in all_recommendations.items():
            if user_id in all_relevant:
                relevant = all_relevant[user_id]
                ap = self.average_precision_at_k(recommended, relevant, k)
                ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def ndcg_at_k(self, recommended_items, relevant_items, k=10):
        """
        Calculate Normalized Discounted Cumulative Gain@K
        
        DCG@K = Σ(rel_i / log2(i+1))
        NDCG@K = DCG@K / IDCG@K
        """
        if len(relevant_items) == 0:
            return 0.0
        
        relevant_set = set(relevant_items)
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because index starts at 0
        
        # Calculate IDCG (ideal DCG)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    # ==================== Comprehensive Evaluation ====================
    
    def evaluate_recommendations(self, user_recommendations, test_ratings, 
                                 k_values=[5, 10, 20], rating_threshold=4.0):
        """
        Comprehensive evaluation of recommendations
        
        Args:
            user_recommendations (dict): {user_id: [recommended_movie_ids]}
            test_ratings (pd.DataFrame): Test ratings
            k_values (list): List of K values to evaluate
            rating_threshold (float): Minimum rating to consider as relevant
            
        Returns:
            dict: Evaluation metrics
        """
        print("\nEvaluating recommendations...")
        
        # Prepare relevant items for each user
        relevant_items = {}
        for user_id in user_recommendations.keys():
            user_test = test_ratings[test_ratings['userId'] == user_id]
            relevant = user_test[
                user_test['rating'] >= rating_threshold
            ]['movieId'].tolist()
            relevant_items[user_id] = relevant
        
        # Calculate metrics for each K
        results = {}
        
        for k in k_values:
            precisions = []
            recalls = []
            f1_scores = []
            aps = []
            ndcgs = []
            
            for user_id, recommended in user_recommendations.items():
                if user_id not in relevant_items:
                    continue
                
                relevant = relevant_items[user_id]
                
                if len(relevant) == 0:
                    continue
                
                # Calculate metrics
                precision = self.precision_at_k(recommended, relevant, k)
                recall = self.recall_at_k(recommended, relevant, k)
                f1 = self.f1_at_k(recommended, relevant, k)
                ap = self.average_precision_at_k(recommended, relevant, k)
                ndcg = self.ndcg_at_k(recommended, relevant, k)
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
                aps.append(ap)
                ndcgs.append(ndcg)
            
            # Calculate averages
            results[f'Precision@{k}'] = np.mean(precisions) if precisions else 0.0
            results[f'Recall@{k}'] = np.mean(recalls) if recalls else 0.0
            results[f'F1@{k}'] = np.mean(f1_scores) if f1_scores else 0.0
            results[f'MAP@{k}'] = np.mean(aps) if aps else 0.0
            results[f'NDCG@{k}'] = np.mean(ndcgs) if ndcgs else 0.0
        
        # Print results
        print("\nEvaluation Results:")
        print("="*60)
        for metric, value in results.items():
            print(f"  {metric:15s}: {value:.4f}")
        print("="*60)
        
        return results
    
    # ==================== Diversity and Coverage ====================
    
    def coverage(self, all_recommendations, all_items):
        """
        Calculate catalog coverage
        
        Coverage = (# of unique recommended items) / (# of all items)
        
        Args:
            all_recommendations (list): List of all recommended items
            all_items (list): List of all available items
            
        Returns:
            float: Coverage score
        """
        unique_recommendations = set(all_recommendations)
        return len(unique_recommendations) / len(all_items)
    
    def diversity(self, recommendations, similarity_matrix):
        """
        Calculate diversity of recommendations
        
        Diversity = 1 - (average pairwise similarity)
        
        Args:
            recommendations (list): List of recommended item IDs
            similarity_matrix (pd.DataFrame): Item similarity matrix
            
        Returns:
            float: Diversity score
        """
        if len(recommendations) < 2:
            return 1.0
        
        similarities = []
        
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                item_i = recommendations[i]
                item_j = recommendations[j]
                
                if item_i in similarity_matrix.index and \
                   item_j in similarity_matrix.columns:
                    sim = similarity_matrix.loc[item_i, item_j]
                    similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        avg_similarity = np.mean(similarities)
        return 1 - avg_similarity
    
    def novelty(self, recommendations, item_popularity):
        """
        Calculate novelty of recommendations
        
        Novelty = -log2(popularity)
        Higher novelty means recommending less popular (more novel) items
        
        Args:
            recommendations (list): List of recommended item IDs
            item_popularity (dict): {item_id: popularity_score}
            
        Returns:
            float: Average novelty score
        """
        novelties = []
        
        for item in recommendations:
            if item in item_popularity:
                pop = item_popularity[item]
                if pop > 0:
                    novelties.append(-np.log2(pop))
        
        return np.mean(novelties) if novelties else 0.0
    
    # ==================== Visualization ====================
    
    def plot_rating_distribution(self, save_path=None):
        """Plot distribution of ratings"""
        plt.figure(figsize=(10, 6))
        
        self.ratings['rating'].value_counts().sort_index().plot(kind='bar')
        plt.title('Distribution of Ratings', fontsize=14, fontweight='bold')
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt
    
    def plot_metrics_comparison(self, metrics_dict, save_path=None):
        """
        Plot comparison of different metrics
        
        Args:
            metrics_dict (dict): Dictionary of metrics from evaluate_recommendations
        """
        # Group metrics by K value
        k_values = sorted(list(set([
            int(k.split('@')[1]) for k in metrics_dict.keys()
        ])))
        
        metric_types = ['Precision', 'Recall', 'F1', 'MAP', 'NDCG']
        
        fig, axes = plt.subplots(1, len(metric_types), figsize=(20, 4))
        
        for idx, metric_type in enumerate(metric_types):
            values = [metrics_dict.get(f'{metric_type}@{k}', 0) for k in k_values]
            
            axes[idx].plot(k_values, values, marker='o', linewidth=2, markersize=8)
            axes[idx].set_title(f'{metric_type}@K', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('K', fontsize=10)
            axes[idx].set_ylabel(metric_type, fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xticks(k_values)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt
    
    def plot_user_activity(self, top_n=20, save_path=None):
        """Plot user activity distribution"""
        user_counts = self.ratings['userId'].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 6))
        user_counts.plot(kind='bar')
        plt.title(f'Top {top_n} Most Active Users', fontsize=14, fontweight='bold')
        plt.xlabel('User ID', fontsize=12)
        plt.ylabel('Number of Ratings', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt
    
    def plot_movie_popularity(self, top_n=20, save_path=None):
        """Plot movie popularity"""
        if self.movies is None:
            print("✗ Movies data not available")
            return None
        
        movie_counts = self.ratings['movieId'].value_counts().head(top_n)
        
        # Get movie titles
        movie_info = self.movies.set_index('movieId')['title'].to_dict()
        labels = [movie_info.get(mid, f'Movie {mid}')[:50] for mid in movie_counts.index]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(movie_counts)), movie_counts.values)
        plt.yticks(range(len(movie_counts)), labels, fontsize=9)
        plt.title(f'Top {top_n} Most Rated Movies', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Ratings', fontsize=12)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt


# Test evaluation
if __name__ == "__main__":
    from data_loader import MovieLensLoader
    
    print("="*60)
    print("Testing Recommender Evaluator")
    print("="*60 + "\n")
    
    # Load data
    loader = MovieLensLoader(data_path='dataset')
    
    if loader.load_data(verbose=False):
        # Create evaluator
        evaluator = RecommenderEvaluator(loader.ratings, loader.movies)
        
        # Test metrics
        print("\n" + "="*60)
        print("Test 1: Calculate basic metrics")
        print("="*60)
        
        recommended = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        relevant = [2, 4, 6, 8, 10, 12, 14]
        
        print(f"Recommended: {recommended}")
        print(f"Relevant: {relevant}")
        print(f"\nPrecision@10: {evaluator.precision_at_k(recommended, relevant, 10):.4f}")
        print(f"Recall@10: {evaluator.recall_at_k(recommended, relevant, 10):.4f}")
        print(f"F1@10: {evaluator.f1_at_k(recommended, relevant, 10):.4f}")
        print(f"AP@10: {evaluator.average_precision_at_k(recommended, relevant, 10):.4f}")
        print(f"NDCG@10: {evaluator.ndcg_at_k(recommended, relevant, 10):.4f}")
        
        print("\n✓ All tests completed!")