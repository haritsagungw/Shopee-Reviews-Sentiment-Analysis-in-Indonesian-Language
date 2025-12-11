# Shopee-Reviews-Sentiment-Analysis-in-Indonesian-Language

## Objective:
To build a sentiment analysis model (positive/negative) for Indonesian-language Shopee reviews.

## Workflow:
1. Data Collection and Cleaning:
- Data Source: Utilizing a Shopee review dataset from Kaggle.
- Data Cleaning: Removing duplicate rows and missing values. Cleaning review text from links, mentions, hashtags, non-alphanumeric characters, and excessive spaces.
2. Text Preprocessing:
- Tokenization: Breaking down text into individual words (tokens).
- Normalization: Using a custom dictionary to normalize slang or abbreviations in Indonesian reviews.
- Stemming: Reducing words to their root form using the Sastrawi library.
- Stopword Removal: Eliminating common words without significant meaning (stopwords) using Sastrawi.
- Output: Cleaned and structured review data ready for sentiment analysis.
3. Sentiment Labeling:
- Method: Employing TextBlob to analyze the sentiment polarity of each review.
- Classification: Assigning 'positive' or 'negative' labels based on the polarity scores.
4. Data Preparation for Modeling:
- Data Splitting: Dividing the dataset into training and testing sets.
- Class Balancing (Resampling): Using SMOTE (Synthetic Minority Over-sampling Technique) on the training data to address class imbalance issues in sentiment categories.
5. Feature Vectorization (TF-IDF):
- Method: Transforming review text into numerical representations using TF-IDF (Term Frequency-Inverse Document Frequency).
- Optimization: Utilizing GridSearchCV to optimize the TfidfVectorizer hyperparameters (such as max_features, ngram_range, min_df, max_df) in conjunction with the classification model.
6. Model Building and Training:
- Model: Employing the Multinomial Naive Bayes classification algorithm.
- Optimization: The model is trained with the best alpha hyperparameter found through GridSearchCV.
7. Model Evaluation:
- Metrics: Using classification_report and accuracy_score to evaluate the model's performance on the test data.
- Results: The classification report shows precision, recall, f1-score, and accuracy for both Positive and Negative sentiments.
8. Model Persistence:
- Serialization: The trained Multinomial Naive Bayes model, TfidfVectorizer, and LabelEncoder are saved using joblib for future reuse without retraining.

## Tools and Libraries Used:
- Data Manipulation: pandas, numpy, re (regular expressions).
- Text Preprocessing: Sastrawi (Stemmer, StopWordRemover), TextBlob (for initial sentiment analysis).
- Machine Learning: scikit-learn (TfidfVectorizer, MultinomialNB, train_test_split, classification_report, accuracy_score, LabelEncoder, GridSearchCV, Pipeline).
- Imbalanced Data Handling: imblearn (SMOTE).
- Model Persistence: joblib.

## Sentiment Analysis Distribution
<img width="609" height="470" alt="1  Sentiment Distribution (Full Processed Dataset by TextBlob)" src="https://github.com/user-attachments/assets/2699d1b0-7ecf-4c13-a764-aaab2bcecff8" />

## Frequently Occurring Words
<img width="661" height="427" alt="2  Word Cloud of Shopee Reviews" src="https://github.com/user-attachments/assets/fdd907ca-0cc1-40f1-9a6a-fa4272a63c22" />

## Confusion Matrix
<img width="640" height="547" alt="3  Confusion Matrix" src="https://github.com/user-attachments/assets/3a1b447e-8616-4755-b32b-0dabac67cb10" />

## Classification Report Heatmap
<img width="674" height="393" alt="4  Classification Report Heatmap (Optimized Model after SMOTE)" src="https://github.com/user-attachments/assets/769dd231-0574-496a-a02a-3fb7a589f6cb" />
