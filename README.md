# 📚 Book Recommender System (Collaborative Filtering & Popularity-Based)

An end-to-end Machine Learning recommendation system built on the **Book-Crossing** dataset. The project combines **popularity-based ranking** (to address cold-start challenges) with **item-based collaborative filtering** using **Cosine Similarity**, deployed through an interactive **Streamlit** web application.

---

## 📌 Project Architecture

```
                             Raw Data (Books, Ratings, Users)
                                           │
                 ┌─────────────────────────┴─────────────────────────┐
                 ▼                                                   ▼
     [Popularity Engine]                                  [Collaborative Filtering]
  - Group by 'Book-Title'                              - Active Readers Filter (> 200 ratings)
  - Threshold: >= 250 reviews                          - Popular Books Filter (>= 50 ratings)
  - Sort by average rating                             - User-Item Interaction Pivot Table
  - Top 50 trending titles                             - Cosine Similarity Metric Matrix
                 │                                                   │
                 ▼                                                   ▼
           popular.pkl                                  pt.pkl, similarity_scores.pkl
                 │                                                   │
                 └─────────────────────────┬─────────────────────────┘
                                           ▼
                                 [Streamlit UI - app.py]
                           Interactive Dual-Engine Dashboard
```

---

## 📊 Dataset Specifications

The project utilizes the standard **Book-Crossing** dataset:
- `Books.csv`: ISBN, Title, Author, Year of Publication, Publisher, Cover Image URLs (`Image-URL-M`).
- `Ratings.csv`: User-ID, ISBN, Book-Rating ($0 - 10$).
- `Users.csv`: User-ID, Location, Age.

---

## ⚙️ Core Recommendation Engines

### 1. Popularity-Based Recommender (Cold-Start Solution)
- **Problem Solved:** When a new user lands on the platform without historical ratings, personal recommendations are impossible.
- **Formulation:** Computes both frequency of ratings (`num_ratings`) and mathematical mean score (`avg_ratings`).
- **Confidence Threshold:** Filters out titles with $< 250$ total ratings to prevent inflated ranks from single $10/10$ reviews, sorting the Top 50 verified titles.

### 2. Item-Based Collaborative Filtering (Personalized Engine)
- **Matrix Sparsity Reduction:**
  - **Active Readers:** Selects only users who have reviewed $> 200$ books (`ratings_with_name.groupby('User-ID').count() > 200`).
  - **Statistically Significant Titles:** Keeps only books reviewed $\ge 50$ times by these power readers.
- **Pivot Table Space:** Rows represent unique book titles; columns represent active user IDs; cell values represent rating magnitude (missing entries filled with `0`).
- **Distance Metric:** Vector angles are calculated using **Cosine Similarity**:
  $$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
- **Top-N Slicing:** For any given title index, queries the top 5 nearest neighbors (excluding self at index 0).

---

## 🗂️ Project Directory Structure

```text
├── Books.csv                # Raw books dataset (ISBN, metadata, URLs)
├── Ratings.csv              # Raw user interaction ratings
├── Users.csv                # Demographic user data
├── main.py                  # Training, preprocessing & serialization script
├── app.py                   # Streamlit web dashboard
├── popular.pkl              # Serialized Top 50 popularity dataframe
├── book_names.pkl           # Serialized index of pivot table titles
├── similarity_scores.pkl    # Serialized pairwise Cosine Similarity matrix
├── books.pkl                # Serialized book metadata & poster URLs
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone Repository & Setup Virtual Environment

```bash
git clone https://github.com/NajeebAhmed69/BOOK_RECOMMENDED_SYSTEM.git
cd book-recommender-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Processing & Serialization

Execute the training script to clean data, reduce sparsity, compute cosine distances, and export `.pkl` files:

```bash
python main.py
```

### 3. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## 📦 Requirements (`requirements.txt`)

```text
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
streamlit>=1.30.0
```

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data Engineering:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (`cosine_similarity`)
- **Persistence:** Pickle
- **Frontend / Deployment:** Streamlit