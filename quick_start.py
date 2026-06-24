# quick_start.py

"""
Quick start script to demonstrate the recommendation system
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.data_loader import MovieLensLoader
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFiltering
from src.matrix_factorization import MatrixFactorization

def main():
    print("="*70)
    print("MOVIE RECOMMENDATION SYSTEM - QUICK START")
    print("="*70 + "\n")
    
    # Load data
    print("📁 Loading dataset...")
    loader = MovieLensLoader(data_path='dataset')
    
    if not loader.load_data(verbose=False):
        print("❌ Failed to load data. Please check the dataset path.")
        return
    
    # Check and sample if dataset is too large
    n_ratings = len(loader.ratings)
    if n_ratings > 1_000_000:
        print(f"\n⚠️  Large dataset detected ({n_ratings:,} ratings)")
        print("   Sampling to smaller size for demo...")
        loader.sample_data(n_users=1000, n_movies=3000, min_ratings_per_user=20)
    
    print(f"✅ Loaded {len(loader.ratings):,} ratings for {len(loader.movies):,} movies\n")
    
    # Get a sample user
    user_id = 1
    print(f"🎬 Getting recommendations for User {user_id}\n")
    
    # Show user's history
    user_movies = loader.get_user_rated_movies(user_id)
    print(f"User {user_id}'s top rated movies:")
    print("-" * 70)
    for idx, row in user_movies.head(5).iterrows():
        print(f"  ⭐ {row['rating']:.1f} - {row['title']}")
    print()
    
    # 1. Content-Based Recommendations
    print("="*70)
    print("1️⃣  CONTENT-BASED RECOMMENDATIONS")
    print("="*70)
    print("Training content-based model...")
    
    try:
        cb = ContentBasedRecommender(loader.movies, loader.ratings, loader.tags)
        cb.compute_similarity()
        
        cb_recs = cb.recommend_for_user(user_id, n=5)
        
        if cb_recs is not None:
            print("\nTop 5 recommendations:")
            for idx, row in cb_recs.iterrows():
                score_col = 'score' if 'score' in row else 'similarity_score'
                print(f"  {idx+1}. {row['title']}")
                print(f"     Genre: {row['genres']} | Score: {row.get(score_col, 0):.2f}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # 2. Matrix Factorization (skip collaborative for large datasets)
    print("="*70)
    print("2️⃣  MATRIX FACTORIZATION (SVD)")
    print("="*70)
    print("Training SVD model (this may take a moment)...")
    
    try:
        mf = MatrixFactorization(loader.ratings, loader.movies)
        mf.train(n_factors=50, n_epochs=10, verbose=False)
        
        # Evaluate
        metrics = mf.evaluate()
        print(f"\nModel Performance:")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        
        mf_recs = mf.recommend_for_user(user_id, n=5)
        
        if mf_recs is not None:
            print("\nTop 5 recommendations:")
            for idx, row in mf_recs.iterrows():
                print(f"  {idx+1}. {row['title']}")
                print(f"     Genre: {row['genres']} | Predicted Rating: {row['predicted_rating']:.2f}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Summary
    print("="*70)
    print("✨ DEMO COMPLETE!")
    print("="*70)
    print("\n💡 For best performance, download the smaller dataset:")
    print("   https://grouplens.org/datasets/movielens/latest/")
    print("   Recommended: ml-latest-small.zip (~1MB, 100K ratings)")
    print("\n🚀 To use the interactive web interface, run:")
    print("   streamlit run app.py")
    print("\n📖 For more information, see README.md")
    print()

if __name__ == "__main__":
    main()