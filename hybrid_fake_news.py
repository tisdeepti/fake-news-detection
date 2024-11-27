import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load datasets
fake_path = r"C:\Users\lilac\OneDrive\Desktop\fake news detection\archive\fake.csv"
true_path = r"C:\Users\lilac\OneDrive\Desktop\fake news detection\archive\true.csv"

fake_data = pd.read_csv(fake_path)
true_data = pd.read_csv(true_path)

# Add class labels
fake_data['class'] = 0  # Fake news
true_data['class'] = 1  # True news

# Combine datasets
data = pd.concat([fake_data, true_data], axis=0).reset_index(drop=True)

# Drop unnecessary columns
data = data.drop(['title', 'subject', 'date'], axis=1)

# Preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"https?://\S+|www\.\S+", '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Apply preprocessing
data['text'] = data['text'].apply(wordopt)

# Use subset of data for faster training
data = data.sample(10000, random_state=42)  # Optimize with 10,000 rows

# Split into features and target
x = data['text']
y = data['class']

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to 5000
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(xv_train, y_train)
lr_predictions = lr_model.predict(xv_test)

print("Logistic Regression")
print(classification_report(y_test, lr_predictions))

# Manual Testing Function
def output_label(n):
    return "Fake News" if n == 0 else "True News"

def manual_testing(news):
    news_data = {"text": [news]}
    df = pd.DataFrame(news_data)
    df['text'] = df['text'].apply(wordopt)
    transformed = vectorizer.transform(df['text'])
    lr_pred = lr_model.predict(transformed)
    
    print("\nManual Testing Results:")
    print(f"Logistic Regression: {output_label(lr_pred[0])}")

# Main Program Loop
def main():
    while True:
        print("\nEnter news text for manual testing:")
        news_text = input()
        manual_testing(news_text)
        
        print("\nDo you want to test another news article? (yes/no):")
        user_choice = input().strip().lower()
        if user_choice != 'yes':
            print("\nExiting the program. Goodbye!")
            break

# Run the main program
if __name__ == "__main__":
    main()
