#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Hybrid Movie Recommendation System
# MovieLens 100K Dataset
# ============================================================

# Step 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

import streamlit as st


# ── Step 2: Load & Preprocess Data ───────────────────────────
BASE_DIR   = os.getcwd()
movies_df  = pd.read_csv(os.path.join(BASE_DIR, "movies.csv"))
ratings_df = pd.read_csv(os.path.join(BASE_DIR, "ratings.csv"))

# Handle missing values
movies_df.dropna(subset=['title', 'genres'], inplace=True)
movies_df = movies_df[movies_df['genres'] != '(no genres listed)']
ratings_df.dropna(subset=['userId', 'movieId', 'rating'], inplace=True)

movies_df  = movies_df.reset_index(drop=True)
ratings_df = ratings_df.reset_index(drop=True)

movies_df.drop_duplicates(subset=['movieId'], inplace=True)
ratings_df.drop_duplicates(inplace=True)


# ── Step 3: Feature Engineering ──────────────────────────────
genres_split  = movies_df['genres'].str.split('|', expand=True)
unique_genres = sorted(set(genres_split.stack()))

for genre in unique_genres:
    movies_df[genre] = movies_df['genres'].apply(
        lambda x: 1 if genre in x.split('|') else 0
    )


# ── Step 4: Merge Datasets ───────────────────────────────────
merged_df = pd.merge(ratings_df, movies_df, on='movieId', how='left')
merged_df.dropna(subset=['title'], inplace=True)
merged_df.reset_index(drop=True, inplace=True)


# ── Step 5: Content-Based Filtering ──────────────────────────
def extract_title_keywords(title):
    title = re.sub(r'\(\d{4}\)', '', title)
    title = re.sub(r'[^a-zA-Z0-9 ]', ' ', title)
    return title.strip().lower()

movies_df['genres_str']    = movies_df['genres'].apply(lambda x: ' '.join(x.split('|')))
movies_df['title_clean']   = movies_df['title'].apply(extract_title_keywords)
movies_df['combined_feat'] = movies_df['genres_str'] + ' ' + movies_df['title_clean']

tfidf        = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['combined_feat'])
cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)
title_to_idx = pd.Series(movies_df.index, index=movies_df['title']).to_dict()


def get_content_based_recs(movie_title, top_n=10):
    if movie_title not in title_to_idx:
        return pd.Series([], dtype=str)
    idx        = title_to_idx[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]


# ── Step 6: Collaborative Filtering (SVD) ────────────────────
reader   = Reader(rating_scale=(0.5, 5.0))
data     = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

svd = SVD(n_factors=100, n_epochs=20, random_state=42)
svd.fit(trainset)

predictions   = svd.test(testset)
surprise_rmse = accuracy.rmse(predictions, verbose=False)


# ── Step 7: Hybrid Recommendation Engine ─────────────────────
scaler_cf = MinMaxScaler(feature_range=(0, 1))


def get_hybrid_recommendations(user_id, movie_title, alpha=0.5, top_n=10):
    if movie_title not in title_to_idx:
        return []

    seed_idx   = title_to_idx[movie_title]
    sim_scores = list(enumerate(cosine_sim[seed_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:51]
    candidates = [(movies_df.iloc[i]['title'], movies_df.iloc[i]['movieId'], score)
                  for i, score in sim_scores]

    hybrid_list = []
    cf_raw      = []
    for title, movie_id, content_score in candidates:
        cf_score = svd.predict(user_id, movie_id).est
        cf_raw.append(cf_score)
        hybrid_list.append({
            'movie_title'  : title,
            'cf_score_raw' : cf_score,
            'content_score': round(content_score, 4),
        })

    cf_arr  = np.array(cf_raw).reshape(-1, 1)
    cf_norm = scaler_cf.fit_transform(cf_arr).flatten()

    for i, item in enumerate(hybrid_list):
        item['cf_score']     = round(cf_norm[i], 4)
        item['hybrid_score'] = round(alpha * cf_norm[i] + (1 - alpha) * item['content_score'], 4)
        del item['cf_score_raw']

    hybrid_list = sorted(hybrid_list, key=lambda x: x['hybrid_score'], reverse=True)[:top_n]
    return hybrid_list


# ── Step 8: Evaluation ───────────────────────────────────────
def evaluate_model(testset_data, svd_model):
    actual_ratings    = [true_r for (_, _, true_r) in testset_data]
    predicted_ratings = [svd_model.predict(uid, iid).est for (uid, iid, _) in testset_data]

    rmse = np.sqrt(np.mean((np.array(actual_ratings) - np.array(predicted_ratings)) ** 2))
    mae  = np.mean(np.abs(np.array(actual_ratings) - np.array(predicted_ratings)))

    actual_bin    = [1 if r >= 3.5 else 0 for r in actual_ratings]
    predicted_bin = [1 if r >= 3.5 else 0 for r in predicted_ratings]

    precision = precision_score(actual_bin, predicted_bin, zero_division=0)
    recall    = recall_score(actual_bin,    predicted_bin, zero_division=0)
    f1        = f1_score(actual_bin,        predicted_bin, zero_division=0)

    return round(rmse, 4), round(mae, 4), round(precision, 4), round(recall, 4), round(f1, 4)


rmse, mae, precision, recall, f1 = evaluate_model(testset, svd)


# ── Step 9: Streamlit Interface ──────────────────────────────
def streamlit_interface():
    st.title("Movie Recommendation System")

    # ── Rating Distribution ──
    st.subheader("Rating Distribution")
    rating_counts = ratings_df['rating'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(rating_counts.index, rating_counts.values, color='steelblue', width=0.4, edgecolor='black')
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Ratings")
    ax.set_title("Distribution of Ratings in Dataset")
    ax.set_xticks(rating_counts.index)
    for i, v in enumerate(rating_counts.values):
        ax.text(rating_counts.index[i], v + 50, str(v), ha='center', fontweight='bold')
    st.pyplot(fig)
    plt.close()

    # ── Evaluation Metrics Chart ──
    st.subheader("Evaluation Metrics")
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(['RMSE', 'MAE'], [rmse, mae], color=['tomato', 'sandybrown'], edgecolor='black')
    axes[0].set_title('Regression Metrics')
    axes[0].set_ylabel('Score (lower is better)')
    for i, v in enumerate([rmse, mae]):
        axes[0].text(i, v + 0.005, f"{v:.4f}", ha='center', fontweight='bold')

    axes[1].bar(['Precision', 'Recall', 'F1-Score'],
                [precision, recall, f1],
                color=['steelblue', 'mediumseagreen', 'mediumpurple'], edgecolor='black')
    axes[1].set_title('Classification Metrics')
    axes[1].set_ylabel('Score (higher is better)')
    axes[1].set_ylim(0, 1.1)
    for i, v in enumerate([precision, recall, f1]):
        axes[1].text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')

    plt.suptitle('Model Evaluation Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Recommendations ──
    user_id     = st.number_input("Enter User ID", min_value=1, max_value=1000, value=1)
    movie_title = st.selectbox("Select Movie Title", movies_df['title'].values)

    if st.button('Get Recommendations'):
        hybrid_recs = get_hybrid_recommendations(user_id, movie_title)

        st.subheader("Collaborative Score for Selected Movie:")
        selected_movie_id = movies_df[movies_df['title'] == movie_title].iloc[0]['movieId']
        st.write(round(svd.predict(user_id, selected_movie_id).est, 4))

        st.subheader("Hybrid Recommendations:")
        recs_df = pd.DataFrame(hybrid_recs)
        st.dataframe(recs_df, use_container_width=True)

        st.subheader("Evaluation Metrics:")
        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'Precision', 'Recall', 'F1-Score'],
            'Value' : [rmse, mae, precision, recall, f1]
        })
        st.dataframe(metrics_df, use_container_width=True)


if __name__ == "__main__":
    streamlit_interface()
