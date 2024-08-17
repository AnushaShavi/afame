import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('spam_csv')

# Display basic information about the dataset
print("Dataset shape:", data.shape)
print("Columns:", data.columns)

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Rename columns for clarity if necessary
# Assuming columns are ['label', 'message']
data.columns = ['label', 'message']

# Encode labels: spam = 1, ham = 0
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Split data into features and labels
X = data['message']
y = data['label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Initialize and train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Save the model for future use
import joblib
joblib.dump(model, 'spam_detector_model.pkl')

# Optional: Save the vectorizer for future use
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
