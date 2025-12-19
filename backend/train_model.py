import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# --- Preprocessing function ---
def clean_text(text):
    text = str(text).lower() # Convert to string and lowercase
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

# --- 1. Load, Clean, and Split Data ---
try:
    # --- Load the data using the correct path ---
    data = pd.read_csv('data/fake_reviews.csv') 
    
    # --- Clean the text data ---
    print("Cleaning text data...")
    
    # *** THIS IS THE LINE I CHANGED FOR YOU ***
    # It now uses your 'text_' column
    data['cleaned_text'] = data['text_'].apply(clean_text)
    
    # --- Define X and y from the data ---
    # These columns must exist in your CSV
    X = data['cleaned_text']  
    y = data['label']         
    
    # --- Split the data ---
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 2. Create and train TF-IDF Vectorizer ---
    print("Training TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # --- 3. Create and train Model ---
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train_tfidf, y_train)

    # --- 4. Save your two files ---
    joblib.dump(model, 'model.pkl')
    joblib.dump(tfidf_vectorizer, 'tfidf.pkl')

    print("\nModels trained and saved successfully!")

except FileNotFoundError:
    print("--- ERROR ---")
    print(f"Error: The file 'data/fake_reviews.csv' was not found.")
    print("Please make sure your 'data' folder and 'fake_reviews.csv' file exist.")

except KeyError:
    print("--- ERROR ---")
    print("Error: Your CSV file is missing a required column.")
    print("Please make sure it has a 'text_' column and a 'label' column.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")