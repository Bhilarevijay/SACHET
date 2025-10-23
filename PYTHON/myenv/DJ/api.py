# import os
# import pandas as pd
# import numpy as np
# import math
# import sys
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from typing import List, Tuple, Dict, Any

# # NOTE: The actual PyTorch/LGBM model definitions (EncoderLSTM, etc.) 
# # MUST remain in predictor.py, as this file imports them.

# # --- CRITICAL: Import your model functions from predictor.py ---
# try:
#     from predictor import (
#         predict_initial_case, 
#         refine_location_with_sightings, 
#         haversine
#     )
    
#     # Importing global data objects required for logic inside this file
#     from predictor import df as raw_df
#     from predictor import CITY_CENTERS as global_city_centers
    
#     # Initialize the prediction objects right after imports
#     # Note: Global models like NLP_MODEL and REFINEMENT_MODEL are 
#     # initialized inside predictor.py and available via its imports.

# except ImportError:
#     # This will still allow the API to load, but endpoints will fail if predictor is missing
#     print("WARNING: Could not import predictor.py. API will not function correctly.")
    
# # --- Pydantic Schemas for API Input/Output ---

# class CaseInput(BaseModel):
#     """Schema for the initial prediction input (matches your app.py inputs)."""
#     child_age: int = Field(..., ge=1, le=18)
#     child_gender: str
#     abduction_time: int = Field(..., ge=0, le=23)
#     abductor_relation: str
#     latitude: float
#     longitude: float
#     day_of_week: int = Field(..., ge=0, le=6)
#     region_type: str
#     population_density: int
#     transport_hub_nearby: int = Field(..., ge=0, le=1) # 1 for Yes, 0 for No

# class Sighting(BaseModel):
#     """Schema for a single sighting update."""
#     lat: float
#     lon: float
#     direction_text: str
#     hours_since: float

# class PredictionUpdate(BaseModel):
#     """Schema for requesting a refined prediction."""
#     # Note: initial_prediction is complex, using dict simplifies schema validation here.
#     initial_prediction: dict 
#     sightings: List[Sighting]
#     initial_case_input: CaseInput

# class PredictionOutput(BaseModel):
#     """Simplified output schema."""
#     risk_label: int
#     risk_prob: float
#     recovered_prob: float
#     recovery_time_hours: float
#     predicted_latitude: float
#     predicted_longitude: float
    
# # --- Initialize FastAPI App ---
# app = FastAPI(
#     title="Sachet ML Prediction API", 
#     description="Backend service for running Stage 1 (Initial Prediction) and Stage 2 (Refinement) ML models.",
#     version="1.0"
# )

# # --- Define Endpoints ---

# @app.get("/")
# def read_root():
#     """Health check endpoint."""
#     return {"status": "ok", "message": "Sachet ML API is running! Access /docs for live testing."}

# @app.post("/predict_initial", response_model=PredictionOutput)
# def initial_prediction_endpoint(input_data: CaseInput):
#     """
#     Stage 1: Generates the initial risk assessment and location hotspot.
#     """
#     try:
#         # 1. Calculate distance to nearest city
#         city_centers = global_city_centers
#         dist = min([haversine(input_data.latitude, input_data.longitude, c_lat, c_lon) 
#                     for c_lat, c_lon in city_centers.values()]) if city_centers else 0
        
#         # 2. Combine inputs for the predictor function
#         case_input_dict = input_data.model_dump()
#         case_input_dict['dist_to_nearest_city'] = dist
        
#         # 3. Run the prediction logic
#         prediction = predict_initial_case(case_input_dict)
        
#         # Convert NumPy float32 to standard float for JSON serialization
#         for key in prediction:
#             if isinstance(prediction[key], np.generic):
#                 prediction[key] = prediction[key].item()
        
#         # Ensure all required keys for PredictionOutput are present
#         return PredictionOutput(**prediction)
    
#     except Exception as e:
#         # Log the error internally and return a generic 500 error to the client
#         print(f"Prediction failed: {str(e)}", file=sys.stderr)
#         raise HTTPException(status_code=500, detail=f"Prediction service failed due to internal model error.")

# @app.post("/refine_location", response_model=Tuple[float, float])
# def refine_location_endpoint(update_data: PredictionUpdate):
#     """
#     Stage 2: Refines the location hotspot based on new sightings (PyTorch model).
#     """
#     try:
#         # Convert Pydantic Sighting models to the dictionary format expected by refine_location_with_sightings
#         sightings_dicts = [s.model_dump() for s in update_data.sightings]
        
#         # Run the actual refinement logic
#         r_lat, r_lon = refine_location_with_sightings(
#             update_data.initial_prediction, 
#             sightings_dicts, 
#             update_data.initial_case_input.model_dump()
#         )
        
#         # Ensure coordinates are standard Python floats
#         return (float(r_lat), float(r_lon))
        
#     except Exception as e:
#         print(f"Refinement failed: {str(e)}", file=sys.stderr)
#         raise HTTPException(status_code=500, detail=f"Refinement service failed due to internal PyTorch error.")

# # --- Helper to expose historical hotspots for the frontend map ---
# @app.get("/historical_hotspots", response_model=List[Tuple[float, float, float]])
# def get_historical_hotspots():
#     """
#     Exposes a simplified list of historical recovery locations for the frontend to cluster/display.
#     The frontend will handle the K-Means logic.
#     Returns: List of (latitude, longitude, risk_level)
#     """
#     # NOTE: The KMeans logic from app.py is too slow for an API, so we just return the raw data
#     # that the frontend can use for map markers.
    
#     if raw_df.empty:
#         return []
    
#     # We will only return a sample of 200 recovered cases for mapping efficiency
#     recovered_df = raw_df[raw_df['recovered']==1].dropna(subset=['recovery_latitude','recovery_longitude'])
    
#     if recovered_df.empty:
#         return []

#     # Sample for performance (adjust as needed)
#     sample_df = recovered_df.sample(n=min(len(recovered_df), 200), random_state=42)














import sys
import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any

# --- CRITICAL: DELAY the heavy import of predictor.py ---
# We will import this module inside the startup event handler.
predictor = None 

# --- Pydantic Schemas (These MUST be defined before the API instance) ---

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
    # FIXED: ge=0 and le=1 ensures value is 0 or 1
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

class PredictionOutput(BaseModel):
    """Simplified output schema."""
    risk_label: int
    risk_prob: float
    recovered_prob: float
    recovery_time_hours: float
    predicted_latitude: float
    predicted_longitude: float

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Sachet ML Prediction API", 
    version="1.0"
)

@app.on_event("startup")
async def startup_event():
    """
    This function runs *after* Uvicorn has started and bound the port.
    It performs the heavy model loading in the background.
    """
    global predictor
    print("\n--- API STARTED: Attempting Delayed Model Loading... ---")
    
    try:
        # 1. Delayed Import: This now triggers the heavy loading in predictor.py
        import predictor as predictor_module 
        global predictor
        predictor = predictor_module
        
        # 2. Trigger the lazy load function inside predictor.py
        #    (We must ensure this function exists and is called correctly)
        predictor._load_models_if_needed() 
        
        print("--- DELAYED MODEL LOADING SUCCESSFUL. API READY. ---")
    except Exception as e:
        print(f"\nFATAL CRASH DURING LAZY LOAD: {e}", file=sys.stderr)
        # Re-raise the exception to terminate the process if loading failed
        sys.exit(1)


# --- Define Endpoints ---

@app.get("/")
def read_root():
    """Health check endpoint."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="API is starting up. Models not yet loaded.")
    return {"status": "ok", "message": "Sachet ML API is running!"}

@app.post("/predict_initial", response_model=PredictionOutput)
def initial_prediction_endpoint(input_data: CaseInput):
    """Stage 1: Generates the initial risk assessment and location hotspot."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Service initializing. Try again in 30 seconds.")
    
    # 1. Calculate distance to nearest city
    city_centers = predictor.CITY_CENTERS
    dist = predictor.haversine(input_data.latitude, input_data.longitude, 0, 0) # Dummy start
    if city_centers:
         dist = min([predictor.haversine(input_data.latitude, input_data.longitude, c_lat, c_lon) 
                    for c_lat, c_lon in city_centers.values()])
    
    case_input_dict = input_data.model_dump()
    case_input_dict['dist_to_nearest_city'] = dist
    
    # 2. Run the prediction logic
    prediction = predictor.predict_initial_case(case_input_dict)
    
    # Convert NumPy float32 to standard float for JSON serialization
    for key in prediction:
        if isinstance(prediction[key], (float, int, bool)): continue
        if isinstance(prediction[key], (list, dict)): continue
        if isinstance(prediction[key], np.generic): prediction[key] = prediction[key].item()
    
    return PredictionOutput(**prediction)


@app.post("/refine_location", response_model=Tuple[float, float])
def refine_location_endpoint(update_data: PredictionUpdate):
    """Stage 2: Refines the location hotspot based on new sightings (PyTorch model)."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Service initializing. Try again in 30 seconds.")
    
    # Run the actual refinement logic
    sightings_dicts = [s.model_dump() for s in update_data.sightings]
    r_lat, r_lon = predictor.refine_location_with_sightings(
        update_data.initial_prediction, 
        sightings_dicts, 
        update_data.initial_case_input.model_dump()
    )
    
    return (float(r_lat), float(r_lon))

@app.get("/historical_hotspots", response_model=List[Tuple[float, float, float]])
def get_historical_hotspots():
    """Exposes historical recovery locations for the frontend map."""
    if predictor is None or predictor.df is None:
        return []
    
    # We will return the raw data and let the frontend handle the display
    recovered_df = predictor.df[predictor.df['recovered']==1].dropna(subset=['recovery_latitude','recovery_longitude'])
    
    if recovered_df.empty:
        return []

    sample_df = recovered_df.sample(n=min(len(recovered_df), 200), random_state=42)

    return list(zip(sample_df['recovery_latitude'], sample_df['recovery_longitude'], [0] * len(sample_df)))


    # Risk level is hard to determine without initial prediction, so we just return coordinates
    return list(zip(sample_df['recovery_latitude'], sample_df['recovery_longitude'], [0] * len(sample_df)))
