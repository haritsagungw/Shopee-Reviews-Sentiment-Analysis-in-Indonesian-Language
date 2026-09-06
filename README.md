# Shopee-Reviews-Sentiment-Analysis-in-Indonesian-Language

An end-to-end NLP sentiment analysis pipeline designed to classify Indonesian e-commerce reviews (Shopee) into **Positif** or **Negatif** sentiments. The project covers raw text cleaning, morphological preprocessing, TF-IDF feature extraction, SMOTE oversampling, and model optimization using Multinomial Naive Bayes.
## Key Features

1. Text Cleaning & Normalization: Strips social media handles, URLs, hashtags, and special characters, followed by slang normalization using custom dictionary mapping.
2. Indonesian NLP Preprocessing: Integrates **Sastrawi** for stemming and stopword removal (with custom rules to preserve negation words like *"tidak"*).
3. Lexicon-Based Auto-Labelling: Applies automated sentiment scoring based on positive and negative keyword lexicons.
4. GridSearchCV Optimization: Tunes TF-IDF vectorizer parameters and `MultinomialNB` hyperparameters (`alpha`) via 5-fold cross-validation.
5. Class Imbalance Handling: Uses **SMOTE** (Synthetic Minority Over-sampling Technique) to rebalance training class distributions.
6. Model Export & Inference: Saves optimized vectorizers and models via `joblib` for easy deployment.
## Performance Summary

| Metric | Score |
| :--- | :--- |
| **Best Cross-Validation Accuracy** | **97.69%** |
| **Test Set Accuracy (Post-SMOTE)** | **96.15%** |
| **Positif Class F1-Score** | **0.98** |
## Quick Start

1. Installation

Clone the repository and install required packages:

```bash
git clone [https://github.com/your-username/shopee-sentiment-analysis.git](https://github.com/your-username/shopee-sentiment-analysis.git)
cd shopee-sentiment-analysis
pip install -r requirements.txt
```
2. Inference Example

Load the trained artifacts and make real-time predictions:
``` python
import joblib

# Load trained models
model = joblib.load('optimized_multinomial_nb_model.joblib')
vectorizer = joblib.load('optimized_tfidf_vectorizer.joblib')
label_encoder = joblib.load('optimized_label_encoder.joblib')

def predict_sentiment(text_list):
    clean_input = [str(t) for t in text_list]
    tfidf_matrix = vectorizer.transform(clean_input)
    prediction = model.predict(tfidf_matrix)
    return label_encoder.inverse_transform(prediction)

# Test sample
review = ["Shopee sangat membantu saya dalam berbelanja"]
result = predict_sentiment(review)
print(f"Review: {review[0]}")
print(f"Sentiment: {result[0]}")
# Output: Sentiment: Positif
```
## Example Output
| Input Review | Predicted Sentiment |
| :--- | :--- |
| `Shopee sangat membantu saya dalam berbelanja` | `Positif` |
