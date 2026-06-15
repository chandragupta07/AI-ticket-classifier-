# Project Context: SupportMind AI

## High-Level Overview
SupportMind AI is an AI-powered customer support ticket classification system. It leverages natural language processing (NLP) and machine learning (ML) to analyze incoming customer support tickets and automatically categorize them into distinct classes such as billing, account/password issues, technical issues, and general inquiries. This helps support teams automate routing, prioritize urgent requests, and reduce response times.

## Architecture
The system follows a classic NLP text classification pipeline:

```
[ Customer Support Ticket Text ]
               │
               ▼
   [ Text Preprocessing ] (Standard lowercasing and cleaning)
               │
               ▼
  [ TF-IDF Vectorization ] (Using vocabulary learned during training)
               │
               ▼
 [ Logistic Regression Model ] (Classifying sparse term vectors)
               │
               ▼
      [ FastAPI Predict API ] (Inference endpoint exposing service)
```

## Tech Stack
- **Language**: Python
- **Framework**: FastAPI (for the high-performance prediction API)
- **ML/NLP Libraries**: Scikit-learn (Logistic Regression, TF-IDF Vectorizer), joblib (serialization)
- **Data Manipulation**: Pandas, Numpy

## Folder Structure
```
AI ticket classifier/
│
├── api/
│   └── main.py              # FastAPI server, schemas, and prediction endpoints
├── data/
│   └── raw/
│       └── tickets.csv      # Generated/downloaded customer support tickets
├── models/
│   ├── tfidf_vectorizer.joblib  # Serialized TF-IDF feature extractor
│   └── ticket_classifier.joblib # Serialized Logistic Regression model
├── notebooks/               # Jupyter Notebooks for experimentation
├── src/
│   ├── generate_data.py     # Script to generate synthetic data
│   ├── prep_twitter_data.py # Script downloading/filtering Twitter support dataset
│   ├── train_and_evaluate.py# Basic model training script
│   └── train_improved.py    # Stratified, balanced, and tuned model training
├── requirements.txt         # Project dependencies
└── README.md                # General readme file
```

## Completed Milestones
1. **Dataset Download & Preparation**: Fetched customer support tweets and prepared an initial dataset with heuristic labels (`prep_twitter_data.py`).
2. **Feature Engineering**: Configured `TfidfVectorizer` with n-grams (1, 2), class-frequency scaling, and vocabulary pruning.
3. **Model Selection & Tuning**: Trained a Logistic Regression classifier with class weights balanced to handle class imbalance (`train_improved.py`).
4. **Serialization**: Saved the trained model and fit vectorizer as reusable artifacts under `models/`.

## Important Engineering Decisions
- **Logistic Regression**: Chosen for its fast inference speeds, low memory footprint, high interpretability, and strong baseline performance on text classification.
- **Separate TF-IDF Vectorizer**: Persisted separately from the model. It *must* be reused during inference to transform raw text into the exact same feature index space mapped during training.
- **FastAPI Lifespan Context Manager**: Used to load the model and vectorizer into memory exactly once at server startup. This avoids reading files from disk on every incoming web request.

## Future Roadmap
- [ ] Complete local verification of the FastAPI prediction server.
- [ ] Add unit tests for preprocessing and endpoint validation.
- [ ] Dockerize the service for containerized deployment.
- [ ] Build a simple frontend dashboard to test tickets interactively.

## Development Rules
- **Educational Mode**: Prioritize clarity. Explain concepts, design decisions, and math step-by-step.
- **Clean modular design**: Avoid monolithic code blocks. Write self-documenting functions with clear inputs and outputs.
- **Zero Placeholders**: Write fully functional demonstration code and real implementations.
