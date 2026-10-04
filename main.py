import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

books = pd.read_csv('Books.csv')
users = pd.read_csv('Users.csv')
ratings = pd.read_csv('Ratings.csv')

# POPULARITY BASED RECOMMENDER SYSTEM
# Merge ratings with book information

ratings_with_name = ratings.merge(books, on='ISBN')

# Number of ratings for each book

num_rating_df = (
    ratings_with_name
    .groupby('Book-Title')
    .count()['Book-Rating']
    .reset_index()
)

num_rating_df.rename(
    columns={'Book-Rating': 'num_ratings'},
    inplace=True
)

# Average rating for each book

avg_rating_df = (
    ratings_with_name
    .groupby('Book-Title')['Book-Rating']
    .mean()
    .reset_index()
)

avg_rating_df.rename(
    columns={'Book-Rating': 'avg_ratings'},
    inplace=True
)

# Combine number of ratings + average rating

popular_df = num_rating_df.merge(
    avg_rating_df,
    on='Book-Title'
)

# Keep books with at least 250 ratings

popular_df = (
    popular_df[
        popular_df['num_ratings'] >= 250
    ]
    .sort_values(
        'avg_ratings',
        ascending=False
    )
    .head(50)
)

# Add book details

popular_df = (
    popular_df
    .merge(books, on='Book-Title')
    .drop_duplicates('Book-Title')
)

popular_df = popular_df[
    [
        'Book-Title',
        'Book-Author',
        'Image-URL-M',
        'num_ratings',
        'avg_ratings'
    ]
]

print("\nPOPULAR BOOKS")
print(popular_df.head())

# COLLABORATIVE FILTERING
# Find users who have rated more than 200 books

x = (
    ratings_with_name
    .groupby('User-ID')
    .count()['Book-Rating']
    > 200
)
# well_read_users = x[x == True].index
well_read_users = x[x].index

# Keep only these users

filtered_rating = ratings_with_name[
    ratings_with_name['User-ID'].isin(well_read_users)
]

# FIND POPULAR BOOKS AMONG THESE USERS

y = (
    filtered_rating
    .groupby('Book-Title')
    .count()['Book-Rating']
    >= 50
)

famous_books = y[y].index

# Keep only famous books

final_ratings = filtered_rating[
    filtered_rating['Book-Title'].isin(famous_books)
]

# CREATE USER-BOOK MATRIX

pt = final_ratings.pivot_table(
    index='Book-Title',
    columns='User-ID',
    values='Book-Rating'
)

# Replace missing ratings with 0

pt.fillna(0, inplace=True)
print("\nPIVOT TABLE")
print(pt.head())

# CALCULATE COSINE SIMILARITY

similarity_scores = cosine_similarity(pt)
print("\nSimilarity matrix shape:")
print(similarity_scores.shape)

# RECOMMENDATION FUNCTION

def recommend(book_name):
    
    # Check if book exists
    
    if book_name not in pt.index:
        print("Book not found in recommendation system.")
        return
    
    # Find index of selected book
    
    index = np.where(pt.index == book_name)[0][0]
    
    # Get similarity scores
    
    similar_books = sorted(
        list(enumerate(similarity_scores[index])),
        key=lambda x: x[1],
        reverse=True
   )[1:6]
    
    print(f"\nRecommendations for: {book_name}\n")
    
    for i in similar_books:
        book_index = i[0]
        recommended_book = pt.index[book_index]
        print(recommended_book)

# TEST RECOMMENDATION
recommend('1984')

# SAVE DATA
pickle.dump(
    popular_df,
    open('popular.pkl', 'wb')
)

pickle.dump(
    pt.index,
    open('book_names.pkl', 'wb')
)

pickle.dump(
    similarity_scores,
    open('similarity_scores.pkl', 'wb')
)

pickle.dump(
    books,
    open('books.pkl', 'wb')
)

print("\nFiles saved successfully!")