import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Function to load and preprocess the data
def load_data(file_path, is_train=True):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ::: ')
            if is_train:
                if len(parts) == 4:
                    data.append((parts[1], parts[2], parts[3]))  # TITLE, GENRE, DESCRIPTION
            else:
                if len(parts) == 3:
                    data.append((parts[1], parts[2]))  # TITLE, DESCRIPTION
    if is_train:
        df = pd.DataFrame(data, columns=['Title', 'Genre', 'Description'])
    else:
        df = pd.DataFrame(data, columns=['Title', 'Description'])
    return df

# Load train and test data
train_data = load_data('/mnt/data/train_data.txt', is_train=True)
test_data = load_data('/mnt/data/test_data.txt', is_train=False)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train = vectorizer.fit_transform(train_data['Title'] + ' ' + train_data['Description'])
y_train = train_data['Genre']

X_test = vectorizer.transform(test_data['Title'] + ' ' + test_data['Description'])

# Split train data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Train a classifier (using Naive Bayes as an example)
model = MultinomialNB()
model.fit(X_train_split, y_train_split)

# Predict on validation data
y_val_pred = model.predict(X_val)

# Evaluate the model
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))

# Predict on the test data
test_predictions = model.predict(X_test)

# Display the test predictions
test_data['Predicted_Genre'] = test_predictions
print(test_data[['Title', 'Predicted_Genre']])

# If needed, save the predictions to a file
test_data[['Title', 'Predicted_Genre']].to_csv('/mnt/data/test_predictions.csv', index=False)
