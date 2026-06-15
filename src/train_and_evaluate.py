import os
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

    # 2. Split the dataset into train and test sets (80% train, 20% test)
    # Using stratify=df['category'] ensures that the train and test sets have the 
    # exact same proportion of each ticket category as the original dataset.
    print("\nStep 2: Splitting dataset into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['category'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['category']
    )
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples:  {len(X_test):,}")

    # 3. Text Vectorization using TF-IDF
    # TF-IDF converts text into numerical vectors that our model can understand.
    print("\nStep 3: Vectorizing text using TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000) # Baseline parameter: limit to top 5,000 words
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Train a Baseline Classifier (Logistic Regression)
    # Logistic Regression is an excellent, fast, and highly interpretable baseline for text classification.
    print("Step 4: Training baseline Logistic Regression classifier...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    # 5. Make Predictions
    print("Step 5: Generating predictions on the test set...")
    y_pred = model.predict(X_test_vec)

    # 6. Generate and Print the Confusion Matrix
    # We will use pandas.crosstab to display the confusion matrix with beautiful row and column headers.
    print("\n" + "="*80)
    print("VISUALIZING THE CONFUSION MATRIX")
    print("="*80)
    
    # Get unique class labels sorted to ensure consistent matrix indexing
    classes = sorted(df['category'].unique())
    
    # Generate the raw confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    # Convert it into a clean, human-readable pandas DataFrame
    cm_df = pd.DataFrame(
        cm, 
        index=[f"Actual: {cls}" for cls in classes], 
        columns=[f"Predicted: {cls}" for cls in classes]
    )
    
    print(cm_df.to_string())
    print("\n" + "="*80)

    # 7. Print the Detailed Classification Report
    print("CLASSIFICATION METRICS (Precision, Recall, F1-Score)")
    print("="*80)
    print(classification_report(y_test, y_pred, target_names=classes))
    print("="*80)

if __name__ == "__main__":
    main()
