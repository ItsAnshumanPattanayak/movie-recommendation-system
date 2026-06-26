# 🎬 Movie Recommendation System 

A comprehensive movie recommendation system implementing multiple state-of-the-art algorithms.

## 📋 Table of Contents
- [Features](#features)
- [Algorithms Implemented](#algorithms-implemented)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Screenshots](#screenshots)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Multiple Recommendation Algorithms**
  - Content-Based Filtering
  - User-User Collaborative Filtering
  - Item-Item Collaborative Filtering
  - Matrix Factorization (SVD)
  - Hybrid Recommendations

- **Interactive Web Interface**
  - Built with Streamlit
  - Real-time recommendations
  - Data visualization and analytics
  - User-friendly UI

- **Comprehensive Evaluation**
  - RMSE & MAE metrics
  - Precision@K & Recall@K
  - Cross-validation
  - Model comparison

## 🎯 Algorithms Implemented

### 1. Content-Based Filtering
Uses movie features (genres, tags) to recommend similar movies.
- **Technique**: TF-IDF Vectorization + Cosine Similarity
- **Best for**: Users with clear preferences

### 2. Collaborative Filtering
Leverages user behavior patterns to make recommendations.
- **User-User**: Finds similar users
- **Item-Item**: Finds similar movies
- **Best for**: Users with rating history

### 3. Matrix Factorization (SVD)
Decomposes user-item matrix to find latent features.
- **Technique**: Singular Value Decomposition
- **Best for**: Accurate rating prediction

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
