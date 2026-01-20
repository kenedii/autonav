# Google Cloud Run Deployment

This directory contains the configuration to deploy the JetRacer Data Frontend and API to Google Cloud Run.

## Architecture
Cloud Run only exposes a single port (default 8080). We use **Nginx** as a reverse proxy within the container to route traffic:
- `/` → Streamlit (Port 8501)
- `/predict`, `/docs`, `/health` → FastAPI (Port 8000)

## Deployment Steps

### 1. Prerequisites
- Google Cloud SDK (`gcloud`) installed and authenticated.
- A Google Cloud Project created.

### 2. Build and Deploy
Run the following commands from the `jetracer/train/data_frontend` directory (one level up from this folder):

```bash
# Set your project ID
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export SERVICE_NAME="jetracer-frontend"

# Submit the build to Cloud Build
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME --file gcp_deploy/Dockerfile .

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2
```

### 3. Accessing the App
- **Streamlit**: Visit the URL provided by Cloud Run (e.g., `https://jetracer-frontend-xyz.a.run.app`).
- **FastAPI**: 
    - Swagger UI: `https://jetracer-frontend-xyz.a.run.app/docs`
    - Prediction Endpoint: `https://jetracer-frontend-xyz.a.run.app/predict`

### 4. Using the App
In the Streamlit "Model Prediction" page:
1. Select "FastAPI Server".
2. Enter the Cloud Run URL + `/predict` as the endpoint (e.g., `https://jetracer-frontend-xyz.a.run.app/predict`).
