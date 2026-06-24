# test_system.py

"""
Test script for Movie Recommendation System
Run this to verify all components are working correctly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import MovieLensLoader
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFiltering
from src.matrix_factorization import MatrixFactorization
from src.evaluation import RecommenderEvaluator

def print_header(text):
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70 + "\n")

def test_data_loader():
    """Test data loading functionality"""
    print_header("TEST 1: Data Loader")
    
    try:
        loader = MovieLensLoader(data_path='dataset')
        assert loader.load_data(verbose=True), "Failed to load data"
        
        # Check dataset size and sample if needed
        n_ratings = len(loader.ratings)
        n_users = loader.ratings['userId'].nunique()
        n_movies = len(loader.movies)
        
        # If dataset is too large (> 1M ratings), sample it
        if n_ratings > 1_000_000:
            print("\n⚠️  Large dataset detected!")
            print("   Sampling to smaller size for testing...")
            loader.sample_data(n_users=1000, n_movies=3000, min_ratings_per_user=20)
        
        # Test methods
        assert loader.ratings is not None, "Ratings not loaded"
        assert loader.movies is not None, "Movies not loaded"
        
        # Test user-item matrix with size check
        print("\nCreating user-item matrix...")
        matrix_size_gb = (loader.ratings['userId'].nunique() * len(loader.movies) * 8) / (1024**3)
        
        if matrix_size_gb < 2:  # Only create if less than 2GB
            loader.create_user_item_matrix()
            assert loader.user_item_matrix is not None, "Failed to create user-item matrix"
        else:
            print(f"⚠️  Skipping dense matrix creation (would be {matrix_size_gb:.2f} GB)")
        
        # Test statistics
        user_stats = loader.get_user_stats()
        assert user_stats is not None, "Failed to get user stats"
        
        movie_stats = loader.get_movie_stats()
        assert movie_stats is not None, "Failed to get movie stats"
        
        print("✅ Data Loader: PASSED")
        return loader
    
    except Exception as e:
        print(f"❌ Data Loader: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return None

def test_content_based(loader):
    """Test content-based recommender"""
    print_header("TEST 2: Content-Based Recommender")
    
    try:
        cb = ContentBasedRecommender(loader.movies, loader.ratings, loader.tags)
        
        # Create features
        cb.create_movie_features()
        assert cb.movie_features is not None, "Failed to create features"
        
        # Compute similarity
        cb.compute_similarity()
        assert cb.similarity_matrix is not None, "Failed to compute similarity"
        
        # Test recommendations
        similar = cb.get_similar_movies(1, n=5)
        assert similar is not None, "Failed to get similar movies"
        assert len(similar) > 0, "No similar movies found"
        
        user_recs = cb.recommend_for_user(1, n=5)
        assert user_recs is not None, "Failed to get user recommendations"
        
        print("✅ Content-Based Recommender: PASSED")
        return cb
    
    except Exception as e:
        print(f"❌ Content-Based Recommender: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return None

def test_collaborative(loader):
    """Test collaborative filtering"""
    print_header("TEST 3: Collaborative Filtering")
    
    try:
        cf = CollaborativeFiltering(loader.ratings, loader.movies)
        
        # Create matrix (will use sparse if needed)
        cf.create_user_item_matrix(use_sparse=True)
        assert cf.user_item_matrix is not None, "Failed to create matrix"
        
        # For large datasets, skip similarity computation
        matrix_size_gb = (loader.ratings['userId'].nunique() * len(loader.movies) * 8) / (1024**3)
        
        if matrix_size_gb < 2:
            # Compute similarities
            cf.compute_user_similarity()
            assert cf.user_similarity is not None, "Failed to compute user similarity"
            
            cf.compute_item_similarity()
            assert cf.item_similarity is not None, "Failed to compute item similarity"
            
            # Test recommendations
            user_recs = cf.user_based_recommend(1, n=5)
            assert user_recs is not None, "Failed to get user-based recommendations"
            
            item_recs = cf.item_based_recommend(1, n=5)
            assert item_recs is not None, "Failed to get item-based recommendations"
        else:
            print("⚠️  Skipping similarity computation for large dataset")
            print("    (Use sampled dataset or implement approximate methods)")
        
        print("✅ Collaborative Filtering: PASSED")
        return cf
    
    except Exception as e:
        print(f"❌ Collaborative Filtering: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return None

def test_matrix_factorization(loader):
    """Test matrix factorization"""
    print_header("TEST 4: Matrix Factorization")
    
    try:
        mf = MatrixFactorization(loader.ratings, loader.movies)
        
        # Prepare data
        mf.prepare_data(test_size=0.2)
        assert mf.trainset is not None, "Failed to prepare trainset"
        assert mf.testset is not None, "Failed to prepare testset"
        
        # Train model with smaller parameters for testing
        mf.train(n_factors=20, n_epochs=5, verbose=False)
        assert mf.model is not None, "Failed to train model"
        
        # Evaluate
        metrics = mf.evaluate()
        assert 'RMSE' in metrics, "RMSE not calculated"
        assert 'MAE' in metrics, "MAE not calculated"
        
        # Test recommendations
        recs = mf.recommend_for_user(1, n=5)
        assert recs is not None, "Failed to get recommendations"
        assert len(recs) > 0, "No recommendations generated"
        
        print("✅ Matrix Factorization: PASSED")
        print(f"   RMSE: {metrics['RMSE']:.4f}")
        print(f"   MAE: {metrics['MAE']:.4f}")
        return mf
    
    except Exception as e:
        print(f"❌ Matrix Factorization: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return None

def test_evaluation(loader):
    """Test evaluation metrics"""
    print_header("TEST 5: Evaluation Metrics")
    
    try:
        evaluator = RecommenderEvaluator(loader.ratings, loader.movies)
        
        # Test metrics
        recommended = [1, 2, 3, 4, 5]
        relevant = [2, 4, 6, 8]
        
        precision = evaluator.precision_at_k(recommended, relevant, 5)
        assert 0 <= precision <= 1, "Invalid precision value"
        
        recall = evaluator.recall_at_k(recommended, relevant, 5)
        assert 0 <= recall <= 1, "Invalid recall value"
        
        f1 = evaluator.f1_at_k(recommended, relevant, 5)
        assert 0 <= f1 <= 1, "Invalid F1 value"
        
        ndcg = evaluator.ndcg_at_k(recommended, relevant, 5)
        assert 0 <= ndcg <= 1, "Invalid NDCG value"
        
        print("✅ Evaluation Metrics: PASSED")
        print(f"   Precision@5: {precision:.4f}")
        print(f"   Recall@5: {recall:.4f}")
        print(f"   F1@5: {f1:.4f}")
        print(f"   NDCG@5: {ndcg:.4f}")
        return evaluator
    
    except Exception as e:
        print(f"❌ Evaluation Metrics: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all tests"""
    print("\n" + "🎬"*35)
    print("MOVIE RECOMMENDATION SYSTEM - TEST SUITE")
    print("🎬"*35)
    
    print("\n💡 TIP: For faster testing, use the MovieLens Small dataset")
    print("   Download from: https://grouplens.org/datasets/movielens/latest/")
    print("   Recommended: ml-latest-small.zip (1MB, ~100K ratings)")
    
    results = {}
    
    # Test 1: Data Loader
    loader = test_data_loader()
    results['Data Loader'] = loader is not None
    
    if loader is None:
        print("\n❌ Cannot proceed without data. Please check your dataset path.")
        print("\n📝 Instructions:")
        print("   1. Download MovieLens dataset")
        print("   2. Extract CSV files to 'dataset/' folder")
        print("   3. Run this script again")
        return
    
    # Test 2: Content-Based
    cb = test_content_based(loader)
    results['Content-Based'] = cb is not None
    
    # Test 3: Collaborative Filtering
    cf = test_collaborative(loader)
    results['Collaborative Filtering'] = cf is not None
    
    # Test 4: Matrix Factorization
    mf = test_matrix_factorization(loader)
    results['Matrix Factorization'] = mf is not None
    
    # Test 5: Evaluation
    evaluator = test_evaluation(loader)
    results['Evaluation'] = evaluator is not None
    
    # Summary
    print_header("TEST SUMMARY")
    
    total = len(results)
    passed = sum(results.values())
    
    for test_name, passed_flag in results.items():
        status = "✅ PASSED" if passed_flag else "❌ FAILED"
        print(f"{test_name:30s}: {status}")
    
    print(f"\n{'='*70}")
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready to use.")
        print("\n🚀 Next steps:")
        print("   - Run 'python quick_start.py' for a demo")
        print("   - Run 'streamlit run app.py' for the web interface")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()