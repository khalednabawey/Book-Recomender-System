# Hybrid Book Recommender System

This is a **Hybrid Recommendation System** deployed on Hugging Face Spaces using **Gradio**.  
It combines **Collaborative Filtering (SVD++)** and **Content-Based Filtering** to recommend books to users, even if they are new to the system (cold-start support).  

🔗 **Live Demo**: [Check it out on Hugging Face Spaces](https://huggingface.co/spaces/khalednabawi11/Book-Recommender-System)  


## Features
```
- Hybrid Model: Combines collaborative filtering and content-based filtering for more accurate recommendations.  
- Cold Start Ready: Recommends top-rated books to new users with no prior ratings.  
- Book Covers: Shows book covers alongside titles for a more visual experience.  
- Gradio: Backend runs on FastAPI, frontend built with Gradio for an interactive experience.  
- Model loaded from Kaggle Models using `kagglehub`.  
```


## How It Works
```
1. For registered users:
   - Predicts ratings for books the user hasn’t rated yet.
   - Shows the top N books sorted by predicted rating.

2. For new users:
   - Recommends globally top-rated books as a fallback (cold start).  

3. Visual Output:
   - Book title, predicted rating, and cover image in a neat table.

```


## Project Structure
```
- ├── app.py # Main Gradio app
- ├── requirements.txt # Dependencies
- ├── README.md
- ├── books-recomendation-system-0.ipynb => Model Development Notebook
```


## Setup Locally

Clone the repo and run:  

```bash
pip install -r requirements.txt
python app.py
```

##  Dependencies

```
- gradio
- pandas
- numpy
- scikit-learn
- kagglehub
- surprise (for SVD++ model)
- fastapi
- uvicorn
```

## Model Details
```
- Collaborative Filtering: SVD++ trained on user-item ratings.
- Content-Based Filtering: Uses book metadata (title, genres, etc.).
- Hybrid approach balances both for better personalization.
```

## Deployment
```
- This app is deployed on Hugging Face Spaces using Gradio UI and loads the model directly from Kaggle Models via kagglehub.
```

