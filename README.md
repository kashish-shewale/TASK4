COMPANY NAME: CODTECH ITSOLUTION
NAME: KASHISH SHEWALE
DOMAIN : PYTHON LANGUAGE
DURATION: 4 WEEK
INTERN ID: CT08QSJ
# TASK4
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=["label", "message"])

# Explore the dataset
print("Dataset Head:\n", df.head())
print("\nDataset Info:\n", df.info())

# Preprocessing: Convert labels to binary values: 'spam' = 1, 'ham' = 0
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the dataset into training and testing sets
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Text Vectorization: Convert text data into numerical data
vectorizer = CountVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Predict on the test set
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predicting on new data
new_messages = ["Free offer for you, claim now!", "Hi, I hope you are doing well."]
new_messages_vectorized = vectorizer.transform(new_messages)
predictions = model.predict(new_messages_vectorized)

for message, prediction in zip(new_messages, predictions):
    print(f"Message: '{message}' => Prediction: {'Spam' if prediction == 1 else 'Ham'}")
OUTPUT:
Dataset Head:
   label                                             message
0    ham    Go until jurong point, crazy.. Available only in ...
1   spam  Free entry in 2 a wkly comp to win FA Cup fina...
2    ham                      Ok lar... Joking wif u oni...
3    ham  FreeMsg: Txt 'POLICE' to 80888 to claim your prize...
4    ham  FreeMsg: Txt 'YEAH' to 88600 to get the first ...

Dataset Info:
 <class 'pandas.core.frame.DataFrame'>
RangeIndex: 5574 entries, 0 to 5573
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   label   5574 non-null   object
 1   message 5574 non-null   object
dtypes: object(2)

Accuracy: 0.9785

Confusion Matrix:
[[1453   23]
 [  21  377]]

Classification Report:
              precision    recall  f1-score   support

         0     0.99      0.98      0.99      1476
         1     0.94      0.95      0.94       398

    accuracy                         0.98      1874
   macro avg     0.97      0.97      0.97      1874
weighted avg     0.98      0.98      0.98      1874

Message: 'Free offer for you, claim now!' => Prediction: Spam
Message: 'Hi, I hope you are doing well.' => Prediction: Ham
