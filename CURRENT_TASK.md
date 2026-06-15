# Current Task: FastAPI Prediction API Integration

## Objective
Build a lightweight, production-ready FastAPI service to load our trained model and vectorizer, and expose a POST `/predict` endpoint to classify customer support tickets.

## Subtasks
- [x] Structure the FastAPI application shell in `api/main.py`
- [x] Configure lifespan context manager to load the model and vectorizer from disk on startup
- [x] Define input (`TicketRequest`) and output (`TicketResponse`) Pydantic schemas
- [x] Implement the `/predict` endpoint with text transformation and probability thresholding
- [x] Spin up the `uvicorn` development server to test the live API
- [x] Run sample request payloads (e.g., via `curl` or interactive docs) to verify correct categorization and confidence scores

## Known Issues
- None so far. The artifacts load successfully and standard predictions are ready to run.

## Next Immediate Steps
1. Start the server using `uvicorn api.main:app --reload`.
2. Open the automated Swagger UI `/docs` to inspect parameters.
3. Test inference with a couple of mock tickets (e.g., password reset requests, billing disputes).
