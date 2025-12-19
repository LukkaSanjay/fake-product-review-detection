import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import time

# --- NEW SELENIUM LIBRARIES ---
from selenium import webdriver
# We no longer need Service or webdriver_manager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# --- BEAUTIFUL SOUP (still needed) ---
from bs4 import BeautifulSoup

# Create the Flask app
app = Flask(__name__)
CORS(app) 

# --- Load the saved models ---
try:
    model = joblib.load('model.pkl')
    tfidf_vectorizer = joblib.load('tfidf.pkl')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    tfidf_vectorizer = None

# --- Preprocessing function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'[^a-z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

# --- *** NEW SCRAPING FUNCTION USING SELENIUM *** ---
def scrape_reviews(url):
    
    if "amazon" not in url and "amzn.in" not in url:
        return None, "This scraper is currently designed for Amazon links only."
            
    print(f"Attempting to scrape URL with Selenium: {url}")

    # Set up Chrome options
    chrome_options = Options()
    
    # --- We are keeping the browser VISIBLE for this test ---
    # chrome_options.add_argument("--headless") 
    
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    # --- *** SIMPLIFIED DRIVER SETUP *** ---
    # Selenium 4.6+ will AUTOMATICALLY download the correct driver (version 141)
    # We no longer need webdriver_manager
    try:
        driver = webdriver.Chrome(options=chrome_options)
    except Exception as e:
        print("--- DRIVER ERROR ---")
        print("Selenium might be having trouble downloading the new driver.")
        print(f"Error: {e}")
        return None, "Error setting up the Chrome Driver. Please ensure you have a good internet connection."
    
    # --- END OF DRIVER SETUP ---

    try:
        driver.get(url)
        
        print("Browser is visible. Waiting 5 seconds for page to load...")
        time.sleep(5) 
        
        html_content = driver.page_source
        driver.quit()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        review_divs = soup.find_all("div", {"data-hook": "review-collapsed"})
        
        reviews = []
        
        if not review_divs:
            print("Selector 'review-collapsed' failed. Trying 'review-text-content'...")
            review_divs = soup.find_all("span", {"class": "review-text-content"})

        for div in review_divs:
            review_text = div.get_text(strip=True)
            if review_text:
                reviews.append(review_text)
        
        print(f"Found {len(reviews)} reviews on the page.")
                
        if not reviews:
            return None, "Could not find any reviews. Amazon is likely blocking the request or has changed its website HTML."
            
        all_reviews_text = " ".join(reviews)
        return all_reviews_text, None

    except Exception as e:
        if 'driver' in locals():
            driver.quit() 
        print(f"Error during Selenium scraping: {e}")
        return None, f"An error occurred: {str(e)}"

# --- Process a list of reviews and get a single prediction ---
def get_prediction(text_block):
    if not text_block:
        return {
            "genuine_percent": 0,
            "fake_percent": 0,
            "opinion": "Error: No text was provided to analyze."
        }
        
    cleaned_review = clean_text(text_block)
    vectorized_review = tfidf_vectorizer.transform([cleaned_review])
    probabilities = model.predict_proba(vectorized_review)[0]
    
    fake_percent = round(probabilities[0] * 100)
    genuine_percent = round(probabilities[1] * 100)
    
    opinion = "Mixed Reviews. Be Cautious (50/50)."
    if genuine_percent > 75:
        opinion = "Looks Good to Buy!"
    elif genuine_percent < 40:
        opinion = "Warning: High Risk of Fake Reviews."

    return {
        "genuine_percent": genuine_percent,
        "fake_percent": fake_percent,
        "opinion": opinion
    }

# --- This is your TEXT prediction endpoint ---
@app.route('/predict', methods=['POST'])
def predict_from_text():
    if not model or not tfidf_vectorizer:
        return jsonify({"error": "Models not loaded"}), 500
    try:
        data = request.get_json()
        review_text = data['review']
        result = get_prediction(review_text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- This is your URL prediction endpoint ---
@app.route('/predict_url', methods=['POST'])
def predict_from_url():
    if not model or not tfidf_vectorizer:
        return jsonify({"error": "Models not loaded"}), 500
    
    try:
        data = request.get_json()
        url = data['url']
        
        scraped_text, error = scrape_reviews(url)
        
        if error:
            return jsonify({"error": error}), 400
        
        result = get_prediction(scraped_text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- Run the app ---
if __name__ == '__main__':
    app.run(debug=True)