# 💬 NLP Sentiment Analysis

A machine learning project that classifies customer reviews as **Positive** or **Negative** using Natural Language Processing (NLP) techniques and multiple machine learning algorithms.

## 🚀 Live Demo

[Try the Sentiment Analysis Application] : https://nlp-sentiment-analysis-yn9na88chbjfv8xyvgmcp8.streamlit.app/

## 📌 Project Overview

This project focuses on analyzing customer reviews from three different sources and predicting their sentiment.

The dataset combines reviews from:
- Amazon
- IMDb
- Yelp

The project includes text preprocessing, TF-IDF feature extraction, model training, performance evaluation, and an interactive Streamlit application.

## 🎯 Objectives

- Collect and combine customer review datasets.
- Clean and preprocess textual data.
- Convert text into numerical features using TF-IDF.
- Train and compare multiple machine learning models.
- Identify the best-performing classification model.
- Deploy an interactive sentiment analysis application.

## 📂 Dataset

The project uses the **UCI Sentiment Labelled Sentences Dataset**.

- Original reviews: 3,000
- Reviews after duplicate removal: 2,982
- Negative reviews: 1,492
- Positive reviews: 1,490

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- TF-IDF
- Streamlit
- Joblib

## ⚙️ Machine Learning Models

Four classification algorithms were trained and evaluated:

1. Naive Bayes
2. Logistic Regression
3. Support Vector Machine (SVM)
4. Random Forest

### 📊 Model Performance

| Model | Test Accuracy |
|---|---:|
| SVM | **81.07%** |
| Naive Bayes | 80.07% |
| Logistic Regression | 79.56% |
| Random Forest | 78.73% |

**Best-performing model:** Support Vector Machine (SVM).

## 🖥️ Application Features

- Enter customer reviews.
- Predict Positive or Negative sentiment.
- View predictions through a simple interactive interface.
- Use the trained SVM model for predictions.

## 📁 Project Structure

```text
NLP-Sentiment-Analysis/
│
├── dataset/
├── notebooks/
├── models/
├── results/
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ▶️ Run Locally

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate to the project folder:

```bash
cd nlp-sentiment-analysis
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run app/app.py
```

## 👩‍💻 Author

**Sayumi Fonseka**

BSc (Hons) in Data Science  
Sri Lanka Institute of Information Technology (SLIIT)

---

⭐ If you find this project interesting, feel free to explore the repository!
