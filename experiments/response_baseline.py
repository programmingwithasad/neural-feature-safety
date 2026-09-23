import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


train_path = "data/processed/response_train.csv"
validation_path = "data/processed/response_validation.csv"
test_path = "data/processed/response_test.csv"

train_data = pd.read_csv(train_path)
validation_data = pd.read_csv(validation_path)
test_data = pd.read_csv(test_path)

train_data = train_data.dropna(
    subset=["response", "response_harm_label"]
)

validation_data = validation_data.dropna(
    subset=["response", "response_harm_label"]
)

test_data = test_data.dropna(
    subset=["response", "response_harm_label"]
)

train_data = train_data[
    train_data["response"].astype(str).str.strip() != ""
]

validation_data = validation_data[
    validation_data["response"].astype(str).str.strip() != ""
]

test_data = test_data[
    test_data["response"].astype(str).str.strip() != ""
]

X_train = train_data["response"].astype(str)
y_train = train_data["response_harm_label"]

X_validation = validation_data["response"].astype(str)
y_validation = validation_data["response_harm_label"]

X_test = test_data["response"].astype(str)
y_test = test_data["response_harm_label"]

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_validation_tfidf = vectorizer.transform(X_validation)
X_test_tfidf = vectorizer.transform(X_test)

print("Training data shape:", X_train_tfidf.shape)
print("Validation data shape:", X_validation_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

validation_predictions = model.predict(X_validation_tfidf)

validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

print("\nValidation Accuracy:", validation_accuracy)

print("\nValidation Classification Report:")
print(
    classification_report(
        y_validation,
        validation_predictions
    )
)

test_predictions = model.predict(X_test_tfidf)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print("\nTest Accuracy:", test_accuracy)

print("\nTest Classification Report:")
print(
    classification_report(
        y_test,
        test_predictions
    )
)