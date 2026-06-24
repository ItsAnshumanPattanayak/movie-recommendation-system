# src/__init__.py

"""
Movie Recommendation System
A comprehensive system for movie recommendations using multiple algorithms.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_loader import MovieLensLoader
from .content_based import ContentBasedRecommender
from .collaborative_filtering import CollaborativeFiltering
from .matrix_factorization import MatrixFactorization
from .evaluation import RecommenderEvaluator

__all__ = [
    'MovieLensLoader',
    'ContentBasedRecommender',
    'CollaborativeFiltering',
    'MatrixFactorization',
    'RecommenderEvaluator'
]