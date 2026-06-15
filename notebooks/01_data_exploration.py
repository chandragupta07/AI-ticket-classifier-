# =============================================================================
# 📘 Step 1: Loading & Exploring the Dataset
# =============================================================================
# As an ML engineer, the VERY FIRST thing you do before building any model
# is understand your data. You never jump into model training blindly.
#
# This script walks you through:
#   1. Loading the dataset with pandas
#   2. Inspecting shape, columns, types, and missing values
#   3. Checking class distribution
#   4. Understanding WHY each step matters
# =============================================================================

# -----------------------------------------------------------------------------
# 1.1  Import pandas — the backbone of data work in Python
# -----------------------------------------------------------------------------
# pandas is a library that gives us "DataFrames" — think of them as
# super-powered spreadsheets that live inside your Python code.
# By convention, we import it as `pd` so every call is short: pd.read_csv(...)

import pandas as pd

# -----------------------------------------------------------------------------
# 1.2  Load the CSV file into a DataFrame
# -----------------------------------------------------------------------------
# pd.read_csv() reads a .csv file and converts it into a DataFrame.
# A DataFrame is a 2D table with rows (samples) and columns (features/labels).
#
# WHY this matters:
#   Your raw data can be in many formats (CSV, JSON, Parquet, SQL, etc.).
#   pandas can read them all. CSV is the most common for tabular ML data.

df = pd.read_csv("data/raw/tickets.csv")
# 'df' is the standard variable name for a DataFrame.
# Think of it as: df = your entire dataset, sitting in memory, ready to explore.

print("✅ Dataset loaded successfully!\n")

# =============================================================================
# 📘 Step 2: Basic Dataset Inspection
# =============================================================================

# -----------------------------------------------------------------------------
# 2.1  View the first few rows — df.head()
# -----------------------------------------------------------------------------
# df.head(n) shows the first `n` rows (default is 5).
#
# WHY this matters:
#   - Gives you an immediate "feel" for the data
#   - You can verify columns are loaded correctly
#   - You can spot obvious problems (weird characters, wrong delimiters, etc.)
#   - It's the fastest sanity check after loading

print("=" * 70)
print("📌 FIRST 5 ROWS (df.head())")
print("=" * 70)
print(df.head())
# Each row = one customer support ticket
# 'text' column   = the actual ticket/tweet content
# 'category' column = the label we want to predict (our TARGET variable)
print()

# -----------------------------------------------------------------------------
# 2.2  Check the shape — df.shape
# -----------------------------------------------------------------------------
# df.shape returns a tuple: (number_of_rows, number_of_columns)
#
# WHY this matters:
#   - Rows = how many samples you have → affects which models you can use
#   - Columns = how many features + labels you have
#   - A dataset with 100 rows vs 100,000 rows needs very different approaches
#   - In NLP, you generally need thousands of samples per class for good results

print("=" * 70)
print("📌 DATASET SHAPE (df.shape)")
print("=" * 70)
rows, cols = df.shape
print(f"Rows (samples): {rows:,}")
print(f"Columns:        {cols}")
print(f"\n→ We have {rows:,} customer support tickets with {cols} columns.\n")

# -----------------------------------------------------------------------------
# 2.3  Inspect column names — df.columns
# -----------------------------------------------------------------------------
# df.columns returns the list of column names in the DataFrame.
#
# WHY this matters:
#   - You need to know EXACTLY what columns exist
#   - Helps identify which column is the INPUT (features) vs OUTPUT (target/label)
#   - Sometimes datasets have extra columns you don't need (IDs, timestamps, etc.)

print("=" * 70)
print("📌 COLUMN NAMES (df.columns)")
print("=" * 70)
print(f"Columns: {df.columns.tolist()}")
print(f"\n→ 'text' = input feature (what the model reads)")
print(f"→ 'category' = target label (what the model predicts)\n")

# -----------------------------------------------------------------------------
# 2.4  Check data types — df.dtypes
# -----------------------------------------------------------------------------
# df.dtypes tells you the data type of each column.
#
# WHY this matters:
#   - ML models expect specific types (numbers, strings, categories)
#   - If a numeric column is loaded as a string, your model will fail
#   - For NLP: text columns should be 'object' (pandas string type)

print("=" * 70)
print("📌 DATA TYPES (df.dtypes)")
print("=" * 70)
print(df.dtypes)
print(f"\n→ Both columns are 'object' type = text strings. ✅ Expected for NLP.\n")

# -----------------------------------------------------------------------------
# 2.5  Check for missing values — df.isnull().sum()
# -----------------------------------------------------------------------------
# df.isnull() returns True/False for every cell (True = missing).
# .sum() counts how many True values per column.
#
# WHY this matters:
#   - Missing values can CRASH your model during training
#   - Even 1 missing value in your text column = error in tokenization
#   - You need to decide: drop missing rows? fill them? ignore them?
#   - This is a critical data quality check BEFORE any processing

print("=" * 70)
print("📌 MISSING VALUES (df.isnull().sum())")
print("=" * 70)
print(df.isnull().sum())

total_missing = df.isnull().sum().sum()
if total_missing == 0:
    print(f"\n→ No missing values! ✅ Our dataset is clean.\n")
else:
    print(f"\n⚠️  Found {total_missing} missing values! We'll need to handle these.\n")

# -----------------------------------------------------------------------------
# 2.6  Quick statistical summary — df.info()
# -----------------------------------------------------------------------------
# df.info() gives a concise summary: column count, types, non-null counts,
# and memory usage — all in one view.
#
# WHY this matters:
#   - It's a one-stop summary combining shape + dtypes + nulls
#   - Memory usage helps you know if the dataset fits in RAM
#   - Professional ML engineers always run this early

print("=" * 70)
print("📌 DATASET INFO (df.info())")
print("=" * 70)
df.info()
print()

# =============================================================================
# 📘 Step 3: Class Distribution — The Single Most Important Check
# =============================================================================

# -----------------------------------------------------------------------------
# 3.1  Count samples per category — df['category'].value_counts()
# -----------------------------------------------------------------------------
# value_counts() counts how many times each unique value appears.
#
# WHY this matters:
#   → This tells you if your dataset is BALANCED or IMBALANCED.
#   → This single check can determine the success or failure of your model!

print("=" * 70)
print("📌 CLASS DISTRIBUTION (df['category'].value_counts())")
print("=" * 70)
class_counts = df['category'].value_counts()
print(class_counts)
print()

# Show percentages too — more intuitive than raw counts
print("📌 CLASS DISTRIBUTION (percentages)")
print("-" * 40)
class_percentages = df['category'].value_counts(normalize=True) * 100
# normalize=True converts counts to proportions (0 to 1)
# Multiply by 100 to get percentages
for category, pct in class_percentages.items():
    print(f"  {category:20s} → {pct:.1f}%")

print()
print("=" * 70)
print("📌 TOTAL UNIQUE CATEGORIES:", df['category'].nunique())
print("=" * 70)

# =============================================================================
# 📘 Step 4: Why Class Imbalance Matters in ML
# =============================================================================
print("""
╔══════════════════════════════════════════════════════════════════════╗
║                  WHY CLASS IMBALANCE MATTERS                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Imagine you have 1000 emails:                                     ║
║    - 950 are NOT spam                                              ║
║    - 50 ARE spam                                                   ║
║                                                                    ║
║  A lazy model that ALWAYS predicts "not spam" gets 95% accuracy!   ║
║  But it catches ZERO spam. That's useless.                         ║
║                                                                    ║
║  This is exactly what happens with IMBALANCED datasets:            ║
║    ❌ The model learns to favor the MAJORITY class                 ║
║    ❌ MINORITY classes get ignored — even though they may be the   ║
║       most important ones (e.g., fraud, disease, urgent tickets)   ║
║    ❌ Accuracy becomes a MISLEADING metric                         ║
║                                                                    ║
║  ─────────────────────────────────────────────────────────────────  ║
║  How do ML engineers handle imbalance?                             ║
║                                                                    ║
║    1. OVERSAMPLING  — duplicate minority class samples             ║
║    2. UNDERSAMPLING — remove majority class samples                ║
║    3. CLASS WEIGHTS  — tell the model to "pay more attention"      ║
║                        to minority classes                         ║
║    4. SMOTE          — generate synthetic minority samples         ║
║    5. BETTER METRICS — use F1-score, precision, recall instead     ║
║                        of accuracy                                 ║
║                                                                    ║
║  We'll address this in a future step when we train our model.      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# 🎯 KEY TAKEAWAYS FROM THIS STEP
# =============================================================================
print("""
🎯 KEY TAKEAWAYS:
─────────────────
1. ALWAYS inspect your data BEFORE building models
2. df.head() → quick visual check of the data
3. df.shape  → how big is the dataset?
4. df.columns → what features/labels do we have?
5. df.isnull().sum() → any missing data to handle?
6. df['target'].value_counts() → is the dataset balanced?
7. Class imbalance is one of the MOST COMMON pitfalls in ML

📍 NEXT STEP: Text preprocessing (cleaning, tokenization, vectorization)
""")
