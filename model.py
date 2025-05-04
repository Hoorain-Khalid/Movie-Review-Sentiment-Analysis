import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Step 1: Load the dataset
data = pd.read_csv('IMDB Dataset.csv')

# Step 2: Preprocess the data
# For simplicity, let's assume the dataset has 'review' and 'sentiment' columns
# Clean and prepare the data (you can modify based on your preprocessing needs)
X = data['review']
y = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)  # Convert sentiments to binary (positive=1, negative=0)

# Step 3: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 6: Save the trained model and vectorizer
pickle.dump((model, vectorizer), open("model.pkl", "wb"))

print("Model training complete and saved as model.pkl")
