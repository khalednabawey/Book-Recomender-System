# import gradio as gr
# import kagglehub
# import pickle
# import numpy as np
# import os
# from surprise import SVDpp
# from sklearn.metrics.pairwise import cosine_similarity

# # --- Download model from KaggleHub ---
# def download_kagglehub_model():
#     user = os.getenv("KAGGLE_USERNAME")         
#     MODEL_SLUG = 'book-recommender-svd' 
#     VARIATION_SLUG = 'v1' 
#     framework = "keras"
    
#     model_handle = f"{user}/{MODEL_SLUG}/{framework}/{VARIATION_SLUG}"
#     print("📥 Downloading model from KaggleHub:", model_handle)
#     model_path = kagglehub.model_download(model_handle)
#     print("✅ Model downloaded to:", model_path)
#     return model_path

# # --- Load Models ---
# def load_models(model_dir):
#     with open(f"{model_dir}/svd_model.pkl", "rb") as f:
#         svdpp_model = pickle.load(f)

#     with open(f"{model_dir}/content_features.pkl", "rb") as f:
#         content_features = pickle.load(f)

#     with open(f"{model_dir}/book_metadata.pkl", "rb") as f:
#         mappings = pickle.load(f)

#     return svdpp_model, content_features, mappings

# # --- Hybrid Prediction ---
# def hybrid_predict(user_id, book_id, alpha=0.7):
#     try:
#     #     uid = user_encoder.transform([user_id])[0]
#     #     iid = item_encoder.transform([book_id])[0]
#         uid = user_id
#         iid = book_id
#     except:
#         return "❌ Unknown user_id or book_id"

#     svd_pred = svdpp_model.predict(uid, iid).est

#     user_liked = np.where(svdpp_model.trainset.ur[uid])[0]
#     if len(user_liked) == 0:
#         content_score = 0
#     else:
#         similarities = cosine_similarity(content_features[iid], content_features[user_liked])
#         content_score = np.mean(similarities)

#     hybrid_score = alpha * svd_pred + (1 - alpha) * content_score * 5
#     return round(hybrid_score, 2)

# # --- Gradio Interface ---
# def recommend(user_id, book_id, alpha=0.7):
#     return f"⭐ Predicted Rating: {hybrid_predict(user_id, book_id, alpha)}"

# # Download and load model
# model_dir = download_kagglehub_model()
# svdpp_model, content_features, mappings = load_models(model_dir)
# # user_encoder = mappings["user_encoder"]
# # item_encoder = mappings["item_encoder"]

# # Start Gradio app
# demo = gr.Interface(
#     fn=recommend,
#     inputs=[
#         gr.Textbox(label="User ID"),
#         gr.Textbox(label="Book ID"),
#         gr.Slider(0, 1, value=0.7, step=0.1, label="Hybrid Weight (alpha)")
#     ],
#     outputs="text",
#     title="📚 Hybrid Book Recommender",
#     description="Enter a user_id and book_id to get a predicted rating using a Hybrid SVD++ and Content-based model."
# )

# demo.launch()

import os
import pickle
import gradio as gr
import pandas as pd
import kagglehub
from surprise import SVDpp, Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# 🔥 Load Hybrid Model from KaggleHub
user = os.getenv("KAGGLE_USERNAME")         
model_slug = 'book-recommender-svd' 
variation_slug = 'v2' 
framework = "keras"

model_handle = f"{user}/{model_slug}/{framework}/{variation_slug}"
model_dir = kagglehub.model_download(model_handle)


with open(f"{model_dir}/svd_model.pkl", "rb") as f:
    svdpp = pickle.load(f)

with open(f"{model_dir}/content_features.pkl", "rb") as f:
    content_features = pickle.load(f)

with open(f"{model_dir}/mappings.pkl", "rb") as f:
    mappings = pickle.load(f)


ratings_df = mappings['ratings_df']
features_df = mappings['feature_df']
user_encoder = mappings['user_encoder']
item_encoder = mappings['item_encoder']

# Create mappings
item_id_to_idx = {bid: idx for idx, bid in enumerate(features_df['book_id'])}
global_mean_rating = ratings_df['rating'].mean()

# User liked books mapping (for content-based filtering)
user_liked_books = ratings_df[ratings_df['rating'] >= 4].groupby('user_id')['book_id'].apply(list).to_dict()

# 🔥 Hybrid Predict Function
def hybrid_predict(user_id, book_id, alpha=0.7):
    try:
        cf_score = svdpp.predict(user_id, book_id).est
    except:
        cf_score = global_mean_rating

    try:
        idx_target = item_id_to_idx[book_id]
        liked_books = user_liked_books.get(user_id, [])
        if not liked_books:
            content_score = global_mean_rating
        else:
            liked_indices = [item_id_to_idx[b] for b in liked_books if b in item_id_to_idx]
            if not liked_indices:
                content_score = global_mean_rating
            else:
                sims = cosine_similarity(content_features[idx_target], content_features[liked_indices])
                content_score = sims.mean()
    except:
        content_score = global_mean_rating

    return alpha * cf_score + (1 - alpha) * content_score


# def recommend_books(user_id, top_n):

#     # uid = user_encoder.transform([user_id])[0]
#     user_id = int(user_id)
    
#     if user_id not in ratings_df['user_id'].unique():
#         return pd.DataFrame([["Error", f"User {user_id} not found."]], columns=["Book Title", "Predicted Rating"])

#     rated_books = set(ratings_df[ratings_df['user_id'] == user_id]['book_id'])
#     unseen_books = set(features_df['book_id']) - rated_books

#     if not unseen_books:
#         return pd.DataFrame([["Error", "No unseen books left to recommend."]], columns=["Book Title", "Predicted Rating"])

#     recommendations = []
#     for book_id in unseen_books:
#         score = hybrid_predict(user_id, book_id)
#         title = features_df.loc[features_df['book_id'] == book_id, 'title'].values[0]
#         recommendations.append((title, round(score, 3)))

#     recommendations.sort(key=lambda x: x[1], reverse=True)
#     return pd.DataFrame(recommendations[:top_n], columns=["Book Title", "Predicted Rating"])


def recommend_books(user_id, top_n):

    user_id = int(user_id)

    if user_id not in ratings_df['user_id'].unique():
        
        # Cold start fallback → Top-N highest-rated books
        top_books = (
            ratings_df.groupby('book_id')['rating']
            .mean()
            .reset_index()
            .sort_values(by='rating', ascending=False)
            .head(top_n)
        )

        recommendations = []
        for _, row in top_books.iterrows():
            book_id = row['book_id']
            avg_rating = row['rating']
            book_info = features_df.loc[features_df['book_id'] == book_id]
            if book_info.empty:
                continue
            title = book_info['title'].values[0]
            image_url = book_info['image_url'].values[0] if 'image_url' in book_info else ""
            recommendations.append(
                (f"<img src='{image_url}' width='60'>", title, round(avg_rating, 3))
            )

        return pd.DataFrame(recommendations, columns=["Book Cover", "Book Title",  "Predicted Rating"])

    rated_books = set(ratings_df[ratings_df['user_id'] == user_id]['book_id'])
    unseen_books = set(features_df['book_id']) - rated_books

    if not unseen_books:
        return pd.DataFrame([["", "No unseen books left to recommend.", ""]], 
                            columns=["Book Cover", "Book Title", "Predicted Rating"])

    recommendations = []
    for book_id in unseen_books:
        score = hybrid_predict(user_id, book_id)
        row = features_df.loc[features_df['book_id'] == book_id]
        title = row['title'].values[0]
        image_url = row['image_url'].values[0] if 'image_url' in row else ""
        recommendations.append((f"<img src='{image_url}' width='60'>", title, round(score, 3)))

    recommendations.sort(key=lambda x: x[2], reverse=True)
    return pd.DataFrame(recommendations[:top_n], columns=["Book Cover", "Book Title", "Predicted Rating"])



# Gradio UI
# with gr.Blocks() as demo:
#     gr.Markdown("## 📚 Hybrid Book Recommendation System")
#     user_id_input = gr.Textbox(label="Enter User ID", placeholder="e.g. 123")
#     top_n_input = gr.Slider(5, 20, value=10, step=1, label="Number of Recommendations")
#     recommend_button = gr.Button("Get Recommendations")
#     output_table = gr.Dataframe(headers=["Book Title", "Predicted Rating"], datatype=["str", "number"])

#     recommend_button.click(
#         recommend_books,
#         inputs=[user_id_input, top_n_input],
#         outputs=output_table
#     )

# demo.launch()

with gr.Blocks() as app_ui:
    gr.Markdown("## 📚 Hybrid Book Recommendation System")
    user_id_input = gr.Textbox(label="User ID", placeholder="Enter user ID")
    # top_n_input = gr.Number(label="Number of Recommendations", value=5, precision=0)
    top_n_input = gr.Slider(5, 20, value=10, step=1, label="Number of Recommendations")
    
    recommend_btn = gr.Button("Recommend Books")
    output = gr.HTML(label="Recommendations")

    recommend_btn.click(
        lambda uid, n: recommend_books(uid, int(n)).to_html(escape=False, index=False),
        inputs=[user_id_input, top_n_input],
        outputs=output
    )


app_ui.launch()

