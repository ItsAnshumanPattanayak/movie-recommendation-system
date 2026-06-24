# debug_content_based.py

from src.data_loader import MovieLensLoader
import pandas as pd

print("Loading data...")
loader = MovieLensLoader(data_path='dataset')
loader.load_data(verbose=False)

print("\n" + "="*60)
print("DEBUGGING CONTENT-BASED ISSUE")
print("="*60)

# Check movies
print(f"\n1. Movies DataFrame:")
print(f"   Shape: {loader.movies.shape}")
print(f"   Columns: {loader.movies.columns.tolist()}")
print(f"   NaN in genres: {loader.movies['genres'].isna().sum()}")

# Check tags
if loader.tags is not None:
    print(f"\n2. Tags DataFrame:")
    print(f"   Shape: {loader.tags.shape}")
    print(f"   Columns: {loader.tags.columns.tolist()}")
    print(f"   NaN in tag column: {loader.tags['tag'].isna().sum()}")
    
    # Show problematic tags
    nan_tags = loader.tags[loader.tags['tag'].isna()]
    if len(nan_tags) > 0:
        print(f"\n   Found {len(nan_tags)} rows with NaN tags")
        print(f"   Sample: {nan_tags.head()}")

# Process genres
print(f"\n3. Processing genres...")
movies_test = loader.movies.copy()
movies_test['genres'] = movies_test['genres'].fillna('Unknown')
movies_test['genres'] = movies_test['genres'].astype(str)
movies_test['features'] = movies_test['genres'].str.replace('|', ' ', regex=False)
print(f"   After processing: {movies_test['features'].isna().sum()} NaN values")
print(f"   Sample features:\n{movies_test['features'].head()}")

# Try TF-IDF on genres only
print(f"\n4. Testing TF-IDF on genres only...")
from sklearn.feature_extraction.text import TfidfVectorizer

features_list = movies_test['features'].tolist()
print(f"   Features list length: {len(features_list)}")
print(f"   Any None values: {None in features_list}")

try:
    tfidf = TfidfVectorizer(max_features=10)
    matrix = tfidf.fit_transform(features_list)
    print(f"   ✓ SUCCESS! Matrix shape: {matrix.shape}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    
    # Find the problematic entry
    for i, feat in enumerate(features_list):
        if feat is None or pd.isna(feat):
            print(f"   Problem at index {i}: {feat}")
            print(f"   Movie: {movies_test.iloc[i]['title']}")

print("\n" + "="*60)
print("Debug complete!")