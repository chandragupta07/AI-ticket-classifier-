import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # 1. Load the dataset
    dataset_path = "data/raw/tickets.csv"
    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset not found at {dataset_path}. Please run prep_twitter_data.py first.")
        return

    print("Step 1: Loading customer support tickets dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df):,} support tickets.")

    # 2. Train-Test Split (with stratification to preserve minority class ratios)
    print("\nStep 2: Splitting dataset into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['category'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['category']
    )

    # 3. Tuning TF-IDF Vectorizer
    # - ngram_range=(1, 2): capture phrases (e.g. "sign in", "charged twice")
    # - min_df=2: ignore ultra-rare terms/typos to reduce noise
    # - sublinear_tf=True: log-scale term frequencies to prevent word stuffing
    print("\nStep 3: Vectorizing text with optimized TF-IDF (N-grams)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), 
        min_df=2, 
        sublinear_tf=True,
        max_features=10000 # Increased feature space to support bigrams
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Train Logistic Regression with Class Weight balancing
    # - class_weight='balanced': weights classes inversely proportional to their frequency
    # - C=2.0: slightly weaker regularization to allow the model to fit minority features better
    print("Step 4: Training improved Logistic Regression with class balancing...")
    model = LogisticRegression(
        max_iter=1000, 
        random_state=42, 
        class_weight='balanced',
        C=2.0
    )
    model.fit(X_train_vec, y_train)

    # 5. Predict
    print("Step 5: Generating predictions on the test set...")
    y_pred = model.predict(X_test_vec)

    # 6. Generate and Print the New Confusion Matrix
    print("\n" + "="*80)
    print("VISUALIZING THE IMPROVED CONFUSION MATRIX")
    print("="*80)
    
    classes = sorted(df['category'].unique())
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    cm_df = pd.DataFrame(
        cm, 
        index=[f"Actual: {cls}" for cls in classes], 
        columns=[f"Predicted: {cls}" for cls in classes]
    )
    
    print(cm_df.to_string())
    print("\n" + "="*80)

    # 7. Print the Detailed Classification Report
    print("IMPROVED CLASSIFICATION METRICS (Precision, Recall, F1-Score)")
    print("="*80)
    print(classification_report(y_test, y_pred, target_names=classes))
    print("="*80)

    # 8. Serialize and Save Model & Vectorizer to disk
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "ticket_classifier.joblib")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    
    print("Step 8: Saving artifacts to disk...")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"Artifacts successfully saved:")
    print(f" - Model:      {model_path}")
    print(f" - Vectorizer: {vectorizer_path}")

if __name__ == "__main__":
    main()
