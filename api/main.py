import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Global variables to store the models and vectorizer in memory
model = None
priority_model = None
sentiment_model = None
vectorizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler to load model and vectorizer on startup,
    and clean up resources on shutdown.
    """
    global model, priority_model, sentiment_model, vectorizer
    
    # Define paths to model artifacts
    model_path = os.path.join("models", "ticket_classifier.joblib")
    priority_model_path = os.path.join("models", "priority_classifier.joblib")
    sentiment_model_path = os.path.join("models", "sentiment_classifier.joblib")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.joblib")
    
    # Check if files exist before trying to load them
    if not all(os.path.exists(p) for p in [model_path, priority_model_path, sentiment_model_path, vectorizer_path]):
        raise FileNotFoundError(
            f"Model artifacts not found. Please ensure they exist at:\n"
            f" - {model_path}\n"
            f" - {priority_model_path}\n"
            f" - {sentiment_model_path}\n"
            f" - {vectorizer_path}"
        )
        
    print("[INFO] Loading model and vectorizer artifacts...")
    # Load serialized objects back into memory
    model = joblib.load(model_path)
    priority_model = joblib.load(priority_model_path)
    sentiment_model = joblib.load(sentiment_model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("[SUCCESS] Models and Vectorizer loaded successfully! Ready for inference.")
    
    yield
    
    # Clean up and release memory on shutdown
    model = None
    priority_model = None
    sentiment_model = None
    vectorizer = None
    print("[CLEANUP] Models and Vectorizer cleared from memory.")

# Initialize the FastAPI app with our custom lifespan handler
app = FastAPI(
    title="AI Support Ticket Classifier",
    description="A production-ready FastAPI service to categorize customer support tickets using Scikit-Learn.",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files directory
app.mount("/static", StaticFiles(directory=os.path.join("api", "static")), name="static")

# --- REQUEST & RESPONSE SCHEMAS ---

class TicketRequest(BaseModel):
    """Schema for incoming prediction requests."""
    text: str = Field(
        ..., 
        description="The raw text of the customer support ticket.",
        examples=["I was charged twice for my subscription this month. Can I get a refund?"]
    )

class TicketResponse(BaseModel):
    """Schema for prediction responses."""
    category: str = Field(..., description="The predicted category of the support ticket.")
    confidence: float = Field(..., description="The model's confidence score (probability) for the category prediction.")
    priority: str = Field(..., description="The predicted priority level of the support ticket.")
    priority_confidence: float = Field(..., description="The model's confidence score (probability) for the priority prediction.")
    sentiment: str = Field(..., description="The predicted sentiment class of the support ticket.")
    sentiment_confidence: float = Field(..., description="The model's confidence score (probability) for the sentiment prediction.")
    reasoning: list[str] = Field(..., description="Top contributing word tokens/n-grams explaining the prediction.")
    suggested_action: str = Field(..., description="Automated recommended operational action for the support team.")

# --- HELPER FUNCTIONS ---

def get_model_explanation(ticket_text: str, vectorizer, model, predicted_class: str) -> list[str]:
    """Extracts top n-grams contributing to the category classification."""
    X_vec = vectorizer.transform([ticket_text])
    non_zero_indices = X_vec.nonzero()[1]
    
    if len(non_zero_indices) == 0:
        return []
        
    class_index = list(model.classes_).index(predicted_class)
    coefficients = model.coef_[class_index]
    feature_names = vectorizer.get_feature_names_out()
    
    contributions = []
    for idx in non_zero_indices:
        weight = float(coefficients[idx] * X_vec[0, idx])
        if weight > 0:
            contributions.append((feature_names[idx], weight))
            
    contributions.sort(key=lambda x: x[1], reverse=True)
    return [term for term, weight in contributions[:3]]

def get_suggested_action(category: str, priority: str, sentiment: str) -> str:
    """Combines predictions to recommend a support action."""
    # Frustrated customer check
    if sentiment == "negative" and priority in ["urgent", "high"]:
        return "CRITICAL: Escalate to Customer Success Escalation Desk immediately."
        
    if category == "billing":
        if priority in ["urgent", "high"]:
            return "Escalate to Billing Tier-2 immediately for transaction review."
        return "Assign to General Billing Queue. SLA target: 8h."
    elif category == "password_reset":
        if priority == "urgent":
            return "Trigger automated 2FA lockout reset sequence. Send email link."
        return "Assign to Identity & Access Management. SLA target: 4h."
    elif category == "technical_issue":
        if priority == "urgent":
            return "Outage warning! Route to Network Operations Team (NOC) immediately."
        return "Route to Tier-1 Technical Support. SLA target: 12h."
    elif category == "general_inquiry":
        return "Route to General Support Queue. SLA target: 24h."
        
    return "Route to Tier-1 Support Queue."

# --- ENDPOINTS ---

@app.get("/health")
def health_check():
    """Health check endpoint to verify the API is alive."""
    return {
        "status": "online",
        "message": "AI Support Ticket Classifier API is running. Head to /docs for interactive documentation."
    }

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the interactive dashboard HTML."""
    html_path = os.path.join("api", "static", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Dashboard UI files not found.")
    return FileResponse(html_path)

@app.post("/predict", response_model=TicketResponse)
def predict(request: TicketRequest):
    """
    Accepts support ticket text, preprocesses and vectorizes it, 
    and predicts the category with a confidence score.
    """
    # 1. Input Validation (Ensure text is not empty or whitespace only)
    ticket_text = request.text.strip()
    if not ticket_text:
        raise HTTPException(
            status_code=400, 
            detail="Input ticket text cannot be empty or whitespace only."
        )
        
    try:
        # 2. Vectorization: Convert raw text to the numerical feature representation
        # NOTE: We use transform(), NOT fit_transform()!
        X_vec = vectorizer.transform([ticket_text])
        
        # 3. Model Inference (Category): Generate classification probability distribution
        probabilities = model.predict_proba(X_vec)[0]
        
        # 4. Category Prediction: Get the class with the highest probability
        predicted_class_index = probabilities.argmax()
        predicted_class = model.classes_[predicted_class_index]
        confidence_score = float(probabilities[predicted_class_index])
        
        # 5. Model Inference (Priority)
        priority_probs = priority_model.predict_proba(X_vec)[0]
        predicted_priority = priority_model.classes_[priority_probs.argmax()]
        priority_confidence_score = float(priority_probs[predicted_priority_index]) if 'predicted_priority_index' in locals() else float(priority_probs[priority_probs.argmax()])
        
        # 6. Model Inference (Sentiment)
        sentiment_probs = sentiment_model.predict_proba(X_vec)[0]
        predicted_sentiment = sentiment_model.classes_[sentiment_probs.argmax()]
        sentiment_confidence_score = float(sentiment_probs[sentiment_probs.argmax()])
        
        # 7. Model Explanation (XAI)
        explanation = get_model_explanation(ticket_text, vectorizer, model, predicted_class)
        
        # 8. Suggested Support Action
        action = get_suggested_action(predicted_class, predicted_priority, predicted_sentiment)
        
        # 9. Return the formatted response
        return TicketResponse(
            category=predicted_class,
            confidence=round(confidence_score, 4),
            priority=predicted_priority,
            priority_confidence=round(priority_confidence_score, 4),
            sentiment=predicted_sentiment,
            sentiment_confidence=round(sentiment_confidence_score, 4),
            reasoning=explanation,
            suggested_action=action
        )
        
    except Exception as e:
        # Catch unexpected errors during inference and return a 500 error
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during inference: {str(e)}"
        )
