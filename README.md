```markdown
# 📚 Book Recommender System (Collaborative Filtering & Popularity-Based)

An end-to-end Machine Learning project that suggests books using a dual-engine architecture: a **Top 50 Popularity-Based Recommender** for new users and an **Item-Based Collaborative Filtering Engine** powered by Cosine Similarity for personalized discovery.

---

## 📌 Project Overview

Recommendation systems are critical for digital libraries and e-commerce platforms to help readers discover titles tailored to their tastes. This project addresses the primary engineering challenges in recommendation systems:

1. **Cold Start Problem:** New users lack reading history. We solve this by serving a curated, high-confidence Top 50 showcase sorted by average ratings.
2. **Extreme Matrix Sparsity:** Large rating matrices often suffer from >95% unrated values. We apply targeted activity thresholds to retain only active users and frequently rated titles.
3. **Item-Based Vector Matching:** Uses vector representations of books across high-dimensional reader spaces to identify similar reading patterns using Cosine Similarity.
4. **Fuzzy Search Assistance:** Includes a case-insensitive fallback search that suggests available titles when an exact title match is not found.

---

## ⚙️ Architecture & Methodology


```

```
                       Raw Book-Crossing Dataset
                     (Books.csv, Ratings.csv, Users.csv)
                                     │
               ┌─────────────────────┴─────────────────────┐
               ▼                                           ▼
   Popularity-Based Engine                     Collaborative Filtering Engine
   ├── Min 250 ratings threshold               ├── Active users (> 50 ratings)
   ├── Sorted by avg_ratings                   ├── Frequent books (>= 10 ratings)
   └── Top 50 showcased titles                 └── Pivot Matrix (Books × Users)
                                                           │
                                                           ▼
                                                Cosine Similarity Matrix
                                                           │
                                                           ▼
                                                Top 5 Similar Recommendations

```

```

### 1. Popularity-Based Engine
* Merges `Ratings.csv` with `Books.csv` by `ISBN`.
* Groups by `Book-Title` to compute both rating frequency (`num_ratings`) and arithmetic mean rating (`avg_ratings`).
* Filters for titles with **$\ge 250$ total ratings** to ensure statistical reliability.
* Ranks descending by `avg_ratings` and slices the top 50 titles, attaching author and cover art thumbnail metadata (`Image-URL-M`).

### 2. Collaborative Filtering Engine
* **Active User Filtering:** Identifies users who submitted **$> 50$ ratings**, isolating experienced reviewers.
* **Book Frequency Filtering:** Retains books that received **$\ge 10$ ratings** from those active users to maintain dense overlap.
* **Pivot Matrix Formulation:** Formulates a user-item matrix where:
  * **Rows:** Book Titles
  * **Columns:** Active User IDs
  * **Values:** Numerical ratings ($1 - 10$), imputing missing values with $0$.
* **Cosine Similarity Calculation:**
  $$\text{Cosine Similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\Vert{}A\Vert{} \Vert{}B\Vert{}} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$
* Recommends the **Top 5** nearest neighbors (ignoring index 0, which is the queried book itself).

---

## 📁 Repository Structure

```text
├── Books.csv                 # Raw book metadata (ISBN, Title, Author, Year, Images)
├── Ratings.csv               # User rating events (User-ID, ISBN, Book-Rating)
├── Users.csv                 # User demographic profiles (User-ID, Location, Age)
├── main.py                   # Data cleaning, pivot generation, similarity & export
├── app.py                    # Interactive Streamlit frontend UI
├── popular.pkl               # Serialized Top 50 popular books DataFrame
├── book_names.pkl            # Serialized list of available book titles (pt.index)
├── similarity_scores.pkl     # Serialized 2D pairwise cosine similarity matrix
├── books.pkl                 # Serialized raw books DataFrame for metadata lookup
├── requirements.txt          # Python project dependencies
└── README.md                 # Project documentation

```

---

## 📦 Pickled Artifacts

When you execute `main.py`, the following four serialized artifacts are saved:

| Artifact File | Contents | Purpose in Application |
| --- | --- | --- |
| `popular.pkl` | Pandas DataFrame | Instant display of Top 50 books without recalculating averages |
| `book_names.pkl` | Index / List | Populates search dropdowns with supported book titles |
| `similarity_scores.pkl` | 2D NumPy Array | Fast $O(1)$ row lookup for nearest neighbors |
| `books.pkl` | Pandas DataFrame | Enriches recommendations with authors and book cover images |

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites & Virtual Environment

Clone the repository and set up a clean Python virtual environment:

```bash
git clone [https://github.com/NajeebAhmed69/BOOK_RECOMMENDED_SYSTEM.git](https://github.com/NajeebAhmed69/BOOK_RECOMMENDED_SYSTEM.git)
cd book-recommender-system

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```

### 2. Install Dependencies

Install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt

```

### 3. Generate Model Artifacts

Execute `main.py` to process the CSV datasets, build the pivot table, and export the `.pkl` models:

```bash
python main.py

```

### 4. Run the Streamlit Web Application

Launch the web interface locally:

```bash
streamlit run app.py

```

Access the application in your browser at `http://localhost:8501`.

---

## 📋 Requirements (`requirements.txt`)

```text
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
streamlit>=1.30.0

```

---

## 👥 Contributors

- **Najeeb Ahmed** - *Machine Learning & Artificial Intelligence Engineer*
  - GitHub: [Najeeb Ahmed](https://github.com/NajeebAhmed69)
  - LinkedIn: [Najeeb Ahmed](www.linkedin.com/in/najeeb-ahmed-346110262)



Contributions, issues, and feature requests are welcome!

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```

<ElicitationsGroup message="Next steps to complete your project repository:">
  <Elicitation label="Create a .gitignore file to exclude heavy datasets and pickles" query="Generate a clean .gitignore file for this Book Recommender project so that large CSVs, virtual environments, and pickle files are properly excluded from Git."/>
  <Elicitation label="Generate an app.py that matches these 4 saved pickle files" query="Provide the full Streamlit app.py code that directly imports popular.pkl, book_names.pkl, similarity_scores.pkl, and books.pkl."/>
  <Elicitation label="Write an MIT LICENSE file for the repository" query="Generate the standard text for an MIT LICENSE file for Najeeb Ahmed."/>
</ElicitationsGroup>

```