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

# Find users who have rated more than 50 books
x = (
    ratings_with_name
    .groupby('User-ID')
    .count()['Book-Rating']
    > 50
)

# Store IDs of active users
active_users = x[x].index

print("Users with > 50 ratings:", len(active_users))

# Keep only ratings given by active users
filtered_rating = ratings_with_name[
    ratings_with_name['User-ID'].isin(active_users)
]

print("Rows after filtering active users:", len(filtered_rating))


# Find books rated by at least 10 active users
y = (
    filtered_rating
    .groupby('Book-Title')
    .count()['Book-Rating']
    >= 10
)

# Store titles of famous/popular books
famous_books = y[y].index

print(
    "Books with >= 10 ratings from active users:",
    len(famous_books)
)

# Keep ratings only for famous books
final_ratings = filtered_rating[
    filtered_rating['Book-Title'].isin(famous_books)
]

print("Rows in final_ratings:", len(final_ratings))

# CREATE BOOK-USER MATRIX

pt = final_ratings.pivot_table(
    index='Book-Title',
    columns='User-ID',
    values='Book-Rating'
)

# Replace missing ratings with 0
pt.fillna(0, inplace=True)

print("\nPIVOT TABLE SHAPE:")
print(pt.shape)

print("\nPIVOT TABLE:")
print(pt.head())


# Check whether the pivot table has data
if pt.empty:
    print(
        "\nPivot table is empty."
        "\nReduce the filtering values."
    )

else:
    # CALCULATE COSINE SIMILARITY
    similarity_scores = cosine_similarity(pt)

    print("\nSimilarity matrix shape:")
    print(similarity_scores.shape)

   # RECOMMENDATION FUNCTION
    def recommend(book_name):

        # Remove extra spaces from user input
        book_name = book_name.strip()

        # Check whether the selected book exists
        if book_name not in pt.index:
            print("\nBook not found in recommendation system.")

            # Find titles containing the entered text
            matching_books = [
                title
                for title in pt.index
                if book_name.lower() in title.lower()
            ]

            # Print possible title matches
            if matching_books:
                print("\nPossible matching books:")

                for title in matching_books[:10]:
                    print(title)

            return

        # Find index of selected book
        index = np.where(pt.index == book_name)[0][0]

        # Find five most similar books
        similar_books = sorted(
            list(enumerate(similarity_scores[index])),
            key=lambda x: x[1],
            reverse=True
        )[1:6]

        print(f"\nRecommendations for: {book_name}\n")

        # Print each recommended book
        for i in similar_books:
            book_index = i[0]
            similarity_score = i[1]

            recommended_book = pt.index[book_index]

            print(
                recommended_book,
                "- Similarity Score:",
                round(similarity_score, 3)
            )
            
    # SEARCH FOR A BOOK TITLE
    search_text = '16 Lighthouse Road'

    matching_books = [
        title
        for title in pt.index
        if search_text.lower() in title.lower()
    ]

    print(f"\nPossible matches for '{search_text}':")

    if len(matching_books) > 0:
        for title in matching_books:
            print(title)
    else:
        print("No matching book found.")


    # TEST RECOMMENDATION SYSTEM

    recommend('16 Lighthouse Road')


    # SAVE FILES

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