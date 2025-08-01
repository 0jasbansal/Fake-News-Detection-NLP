import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

print("--- Fake News Detector ---")
print("\n[+] STEP 1: Loading datasets...")
try:
    df_true = pd.read_csv('True.csv')
    df_fake = pd.read_csv('Fake.csv')
    print("...datasets loaded successfully.")
except FileNotFoundError:
    print("\n*** ERROR: Make sure 'True.csv' and 'Fake.csv' are in the same folder. ***")
    exit()

df_true['label'] = 0
df_fake['label'] = 1

df = pd.concat([df_true[['title', 'label']], df_fake[['title', 'label']]])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df['title']
y = df['label']
print(f"Dataset prepared. Total articles: {len(y)}")

print("\n[+] STEP 2: Cleaning text data...")
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))
corpus = []
for i in range(len(X)):
    review = re.sub('[^a-zA-Z]', ' ', X.iloc[i])
    review = review.lower()
    review = review.split()
    review = [port_stem.stem(word) for word in review if not word in stop_words]
    review = ' '.join(review)
    corpus.append(review)
print("...text cleaning complete.")

print("\n[+] STEP 3: Converting text to numbers...")
tfidf_v = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
X_features = tfidf_v.fit_transform(corpus).toarray()
print("...text converted.")

print("\n[+] STEP 4: Training the model...")
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(X_train, y_train)
print("...model trained.")

print("\n[+] STEP 5: Evaluating model...")
y_pred = pac.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {score*100:.2f}%')
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print("\n--- Done ---")
