import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data
X = [
    "I love this product! It's amazing!",
    "Terrible experience, I hate it.",
    "The weather is beautiful today.",
    "I can't believe how bad this is.",
    "This movie is fantastic.",
    "I'm not a fan of this restaurant.",
    "Great news! I got a promotion.",
    "I feel so disappointed.",
    "I'm having a great day!",
    "This book is a masterpiece.",
    "I'm so excited about the concert.",
    "Worst service ever!",
    "Delicious food at this restaurant.",
    "This place is a hidden gem.",
    "I can't stand this traffic.",
    "I'm feeling very optimistic today.",
    "What a waste of time!",
    "I'm really impressed by their service.",
    "I can't get enough of this song.",
    "The movie was a letdown.",
    "This event exceeded my expectations.",
    "I'm tired of this routine.",
    "The view from the mountain is breathtaking."
]

# Labels: 1 for positive sentiment, 0 for negative sentiment
y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transforming text data into TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Training a Logistic Regression model
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train_tfidf, y_train)

# Predicting sentiment for test data
y_pred = logistic_regression.predict(X_test_tfidf)

# Evaluating the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)

# Predicting sentiment for a new text
new_text = ["The view from the mountain is breathtaking."]
new_text_tfidf = tfidf_vectorizer.transform(new_text)
prediction = logistic_regression.predict(new_text_tfidf)

if prediction[0] == 1:
    print("Positive sentiment")
else:
    print("Negative sentiment")
