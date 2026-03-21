import kagglehub
import pandas as pd
import os
import re

def categorize_tweet(text):
    """
    Heuristic rule-based function to assign an initial category 
    so we have 'ground truth' labels to train our ML model.
    """
    text = str(text).lower()
    
    # Define keywords for each category
    billing_keywords = ['charge', 'bill', 'money', 'refund', 'pay', 'invoice', 'subscription', 'cancel', 'dollar', 'euro', 'price']
    tech_keywords = ['crash', 'error', 'bug', 'glitch', 'down', 'slow', 'load', 'frozen', 'stuck', 'connection', 'offline']
    password_keywords = ['password', 'login', 'reset', 'account locked', 'sign in', 'can\'t access', 'auth']
    
    if any(word in text for word in password_keywords):
        return 'password_reset'
    elif any(word in text for word in tech_keywords):
        return 'technical_issue'
    elif any(word in text for word in billing_keywords):
        return 'billing'
    else:
        return 'general_inquiry'

def main():
    print("1. Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
    print("Dataset stored at:", path)
    
    # Locate the CSV file in the downloaded path
    csv_file = os.path.join(path, "twcs", "twcs.csv")
    if not os.path.exists(csv_file):
        # depending on extract structure
        csv_file = os.path.join(path, "twcs.csv")
    
    print(f"2. Loading CSV file from {csv_file}...")
    df = pd.read_csv(csv_file, nrows=50000) # Load subset to avoid massive memory usage for now
    
    # 3. We only want inbound tweets (from customers, not from support agents)
    customers_df = df[df['inbound'] == True].copy()
    
    print(f"Loaded {len(customers_df)} customer tweets. Categorizing now...")
    
    # 4. Apply our categorization logic
    customers_df['category'] = customers_df['text'].apply(categorize_tweet)
    
    # Keep only the relevant columns and rename for our pipeline
    final_df = customers_df[['text', 'category']]
    
    target_path = "data/raw/tickets.csv"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    final_df.to_csv(target_path, index=False)
    
    print(f"✅ Preprocessing complete! Saved {len(final_df)} categorized tickets to {target_path}")
    print("\nDataset Category Distribution:")
    print(final_df['category'].value_counts())
    
    print("\nSample Data:")
    print(final_df.head(10))

if __name__ == "__main__":
    main()
