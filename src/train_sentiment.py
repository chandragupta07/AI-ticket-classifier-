import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # 1. Load the dataset
    dataset_path = "data/raw/tickets.csv"
    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset not found at {dataset_path}. Please run generate_data.py first.")
        return

    print("Step 1: Loading customer support tickets dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df):,} support tickets.")

    # 2. Train-Test Split (with stratification on sentiment)
    print("\nStep 2: Splitting dataset into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['sentiment'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['sentiment']
    )

    # 3. Load pre-fit Vectorizer (Reusing the category TF-IDF vocabulary pipeline)
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.joblib")
    if not os.path.exists(vectorizer_path):
        print(f"[Error] TF-IDF Vectorizer not found at {vectorizer_path}. Run train_improved.py first.")
        return
        
    print(f"\nStep 3: Loading pre-fit TF-IDF Vectorizer from {vectorizer_path}...")
    vectorizer = joblib.load(vectorizer_path)
    
    print("Vectorizing train and test sets...")
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Train Logistic Regression with Class Weight balancing
    print("Step 4: Training Logistic Regression sentiment classifier...")
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

    # 6. Generate and Print the Confusion Matrix
    print("\n" + "="*80)
    print("VISUALIZING THE SENTIMENT CONFUSION MATRIX")
    print("="*80)
    
    classes = sorted(df['sentiment'].unique())
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    cm_df = pd.DataFrame(
        cm, 
        index=[f"Actual: {cls}" for cls in classes], 
        columns=[f"Predicted: {cls}" for cls in classes]
    )
    
    print(cm_df.to_string())
    print("\n" + "="*80)

    # 7. Print the Detailed Classification Report
    print("SENTIMENT CLASSIFICATION METRICS (Precision, Recall, F1-Score)")
    print("="*80)
    print(classification_report(y_test, y_pred, target_names=classes))
    print("="*80)

    # 8. Serialize and Save Sentiment Model
    models_dir = "models"
    model_path = os.path.join(models_dir, "sentiment_classifier.joblib")
    
    print("Step 8: Saving Sentiment Model to disk...")
    joblib.dump(model, model_path)
    print(f"Sentiment classifier successfully saved to: {model_path}")

if __name__ == "__main__":
    main()
