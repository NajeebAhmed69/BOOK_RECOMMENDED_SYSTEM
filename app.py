import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Book Recommender System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .book-card {
        padding: 10px;
        border-radius: 8px;
        background-color: #f8f9fa;
        margin-bottom: 20px;
        min-height: 420px;
    }
    .book-title {
        font-size: 15px;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 4px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .book-author {
        font-size: 13px;
        color: #6c757d;
        margin-bottom: 4px;
    }
    .book-metric {
        font-size: 12px;
        color: #28a745;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_artifacts():
    with open('popular.pkl', 'rb') as f:
        popular_df = pickle.load(f)
    with open('book_names.pkl', 'rb') as f:
        book_names = pickle.load(f)
    with open('similarity_scores.pkl', 'rb') as f:
        similarity_scores = pickle.load(f)
    with open('books.pkl', 'rb') as f:
        books_df = pickle.load(f)
    return popular_df, book_names, similarity_scores, books_df

try:
    popular_df, book_names, similarity_scores, books_df = load_artifacts()
    # Prepare clean metadata reference
    books_meta = books_df.drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M']]
except FileNotFoundError as e:
    st.error("Model artifacts not found! Ensure 'popular.pkl', 'book_names.pkl', 'similarity_scores.pkl', and 'books.pkl' exist in the root folder.")
    st.stop()
def get_recommendations(selected_book_title):
    if selected_book_title not in book_names:
        return []

    # Find row index corresponding to selected book
    book_idx = np.where(book_names == selected_book_title)[0][0]

    # Sort similarity scores descending (excluding self at index 0)
    similar_indices = sorted(
        list(enumerate(similarity_scores[book_idx])),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommendation_cards = []
    for item in similar_indices:
        idx = item[0]
        score = item[1]
        rec_title = book_names[idx]
        
        # Fetch metadata
        meta = books_meta[books_meta['Book-Title'] == rec_title]
        if not meta.empty:
            author = meta['Book-Author'].values[0]
            img_url = meta['Image-URL-M'].values[0]
        else:
            author = "Unknown Author"
            img_url = "https://via.placeholder.com/150x220?text=No+Cover"

        recommendation_cards.append({
            "title": rec_title,
            "author": author,
            "image": img_url,
            "match": f"{score * 100:.1f}%"
        })

    return recommendation_cards

st.title("Book Recommender System")
st.caption("Powered by Collaborative Filtering (Cosine Similarity) & Popularity Metrics")

tab1, tab2 = st.tabs(["Personalized Recommendations", "Top 50 Popular Books"])

with tab1:
    st.subheader("Recommend by Book Title")
    st.markdown("Select a title to retrieve similar books based on community reading profiles:")

    # Selectbox fed directly from book_names.pkl
    selected_book = st.selectbox(
        "Search or choose a book title:",
        options=sorted(list(book_names)),
        index=0
    )

    if st.button("Generate Recommendations", type="primary", use_container_width=True):
        recs = get_recommendations(selected_book)

        if recs:
            st.write(f"### Top 5 Recommendations for: *{selected_book}*")
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    # Fallback on broken image links
                    img_src = recs[i]['image'] if str(recs[i]['image']).startswith('http') else "https://via.placeholder.com/150x220?text=No+Cover"
                    st.image(img_src, use_container_width=True)
                    st.markdown(f"**{recs[i]['title']}**")
                    st.caption(f"{recs[i]['author']}")
                    st.success(f"Similarity: {recs[i]['match']}")
        else:
            st.warning("No recommendations available for the selected title.")
with tab2:
    st.subheader("Top 50 Trending Books")
    st.caption("Highest rated books with a minimum of 250 user reviews.")

    # Render popular_df in a grid of 5 items per row
    num_books = len(popular_df)
    items_per_row = 5
    for r in range(0, num_books, items_per_row):
        row_cols = st.columns(items_per_row)
        for c in range(items_per_row):
            idx = r + c
            if idx < num_books:
                item = popular_df.iloc[idx]
                with row_cols[c]:
                    img_src = item['Image-URL-M'] if str(item['Image-URL-M']).startswith('http') else "https://via.placeholder.com/150x220?text=No+Cover"
                    st.image(img_src, use_container_width=True)
                    st.markdown(f"**{item['Book-Title']}**")
                    st.caption(f"{item['Book-Author']}")
                    st.info(f"{item['avg_ratings']:.2f} ({int(item['num_ratings'])} votes)")