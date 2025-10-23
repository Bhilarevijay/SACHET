import os
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any

# --- CRITICAL: Import your model functions ---
try:
    from predictor import predict_initial_case, refine_location_with_sightings, haversine
except ImportError:
    # If this fails, the app will start but raise an error on API calls
    raise HTTPException(status_code=500, detail="FATAL: Could not load predictor.py. Check deployment files.")

# --- Configuration and Data Loading ---
RANDOM_SEED = 42
DATASET_PATH = "sachet_main_cases_2M.csv" 
CITIES_DATA_PATH = "worldcities.csv"     

# In-memory global variables for data/model access
df = pd.DataFrame()
cities_df = pd.DataFrame()
CITY_CENTERS = {}

def load_global_data():
    """Loads CSV files once when the API starts."""
    global df, cities_df, CITY_CENTERS
    
    # NOTE: Assuming files are small enough for Render's memory tier
    df = pd.read_csv(DATASET_PATH, usecols=['abduction_time','abductor_relation','region_type','recovered','recovery_latitude','recovery_longitude'])
    cities_df = pd.read_csv(CITIES_DATA_PATH)
    
    major_cities = cities_df[cities_df['population'] > 500000]
    CITY_CENTERS = {row['city']: (row['lat'], row['lng']) for index, row in major_cities.iterrows()}
    
    if df.empty or cities_df.empty:
        raise Exception(f"CRITICAL: Failed to load required data. Check file paths: {DATASET_PATH}")
    print("--- Data Loaded Successfully ---")

# --- Pydantic Schemas for Input/Output ---

class CaseInput(BaseModel):
    """Schema for the initial prediction input."""
    child_age: int = Field(..., ge=1, le=18)
    child_gender: str
    abduction_time: int = Field(..., ge=0, le=23)
    abductor_relation: str
    latitude: float
    longitude: float
    day_of_week: int = Field(..., ge=0, le=6)
    region_type: str
    population_density: int
    transport_hub_nearby: int = Field(..., ge=0, le=1)

class Sighting(BaseModel):
    """Schema for a single sighting update."""
    lat: float
    lon: float
    direction_text: str
    hours_since: float

class PredictionUpdate(BaseModel):
    """Schema for requesting a refined prediction."""
    initial_prediction: dict
    sightings: List[Sighting]
    initial_case_input: CaseInput

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Sachet ML Prediction API", 
    version="1.0",
    on_startup=[load_global_data] # Load data when the server starts
)

# --- Endpoints ---

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Sachet ML API is running!"}

@app.post("/predict_initial", response_model=Dict[str, Any])
def initial_prediction_endpoint(input_data: CaseInput):
    """
    Stage 1: Generates the initial risk assessment and location hotspot.
    """
    try:
        # 1. Calculate distance to nearest city (using global CITY_CENTERS)
        if not CITY_CENTERS:
            dist = 0
        else:
            dist = min([haversine(input_data.latitude, input_data.longitude, c_lat, c_lon) 
                        for c_lat, c_lon in CITY_CENTERS.values()])
        
        case_input_dict = input_data.model_dump()
        case_input_dict['dist_to_nearest_city'] = dist
        
        # 2. Run the actual prediction logic
        prediction = predict_initial_case(case_input_dict)
        
        # Add the full input back for the frontend to store
        prediction['initial_case_input'] = case_input_dict
        
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/refine_location", response_model=Tuple[float, float])
def refine_location_endpoint(update_data: PredictionUpdate):
    """
    Stage 2: Refines the location hotspot based on new sightings.
    """
    try:
        # Convert Pydantic Sighting models to the dictionary format expected by your function
        sightings_dicts = [s.model_dump() for s in update_data.sightings]
        
        # Run the actual refinement logic
        r_lat, r_lon = refine_location_with_sightings(
            update_data.initial_prediction, 
            sightings_dicts, 
            update_data.initial_case_input.model_dump()
        )
        return (r_lat, r_lon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refinement failed: {str(e)}")

# --- End of api.py ---
