# import joblib
# import os
# import math
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import torch
# import torch.nn as nn
# import sys

# # --- CONFIGURATION & ROBUST PATHING ---
# # Get the absolute path of the directory where this script is located
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR_STG1 = os.path.join(SCRIPT_DIR, "models_lgbm_tuned")
# MODEL_DIR_STG2_PYTORCH = os.path.join(SCRIPT_DIR, "models_refinement_pytorch")
# MAX_SEQ_LEN = 5  # Must match the training script

# # --- PYTORCH MODEL DEFINITIONS (THE DEFINITIVE SEQ2SEQ ARCHITECTURE) ---
# # This block is an exact, perfect match of the architecture in train.py
# class EncoderLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size=128, num_layers=2):
#         super(EncoderLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
#     def forward(self, x):
#         _, (hidden, cell) = self.lstm(x)
#         return hidden, cell

# class DecoderLSTM(nn.Module):
#     def __init__(self, output_size, hidden_size=128, num_layers=2):
#         super(DecoderLSTM, self).__init__()
#         self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
#         self.fc_out = nn.Linear(hidden_size, output_size)
#     def forward(self, x, hidden, cell):
#         output, (hidden, cell) = self.lstm(x, (hidden, cell))
#         prediction = self.fc_out(output)
#         return prediction, hidden, cell

# class RefinementEngine(nn.Module):
#     def __init__(self, encoder, decoder, device):
#         super(RefinementEngine, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.device = device
#     def forward(self, source, target_len=1):
#         batch_size = source.shape[0]
#         outputs = torch.zeros(batch_size, target_len, 2).to(self.device)
#         hidden, cell = self.encoder(source)
#         # Use the last known sighting's lat/lon as the initial input for the decoder
#         x = source[:, -1, 0:2].unsqueeze(1)
#         for t in range(target_len):
#             output, hidden, cell = self.decoder(x, hidden, cell)
#             outputs[:, t, :] = output.squeeze(1)
#             x = output # Use the predicted output as the next input
#         return outputs

# # --- GLOBAL MODEL LOADING ---
# try:
#     print("--- Sachet AI Engine Initializing: Loading all models... ---")
    
#     # Load Stage 1 (LightGBM) Models
#     print(f"Attempting to load Stage 1 models from: {MODEL_DIR_STG1}")
#     PIPELINE_STG1 = joblib.load(os.path.join(MODEL_DIR_STG1, 'pipeline.joblib'))
#     CLF_RISK = joblib.load(os.path.join(MODEL_DIR_STG1, 'clf_risk.joblib'))
#     CLF_RECOVERED = joblib.load(os.path.join(MODEL_DIR_STG1, 'clf_recovered.joblib'))
#     REG_TIME = joblib.load(os.path.join(MODEL_DIR_STG1, 'reg_recovery_time.joblib'))
#     REG_LAT = joblib.load(os.path.join(MODEL_DIR_STG1, 'reg_recovery_lat.joblib'))
#     REG_LON = joblib.load(os.path.join(MODEL_DIR_STG1, 'reg_recovery_lon.joblib'))
#     print("Stage 1 (LightGBM) Models loaded successfully.")

#     # Load Stage 2 (PyTorch Trajectory) Models
#     print(f"Attempting to load Stage 2 models from: {MODEL_DIR_STG2_PYTORCH}")
#     SEQ_FEATURE_SIZE = 3 + 384  # 3 for lat/lon/time, 384 for embedding size
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Instantiate the correct model architecture before loading
#     encoder_model = EncoderLSTM(SEQ_FEATURE_SIZE).to(DEVICE)
#     decoder_model = DecoderLSTM(output_size=2).to(DEVICE)
#     REFINEMENT_MODEL = RefinementEngine(encoder_model, decoder_model, DEVICE).to(DEVICE)
    
#     REFINEMENT_MODEL.load_state_dict(torch.load(os.path.join(MODEL_DIR_STG2_PYTORCH, 'refinement_model.pth'), map_location=torch.device(DEVICE)))
#     REFINEMENT_MODEL.eval() # Set model to evaluation mode
#     print("Stage 2 (PyTorch) Refinement Engine loaded successfully.")
    
#     # Load NLP Model
#     print("Loading NLP Sentence Transformer...")
#     NLP_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    
#     print("\n--- All prediction models loaded successfully. AI Engine is ready. ---")

# except FileNotFoundError as e:
#     print("\n" + "="*80); print("FATAL ERROR: A required model file was not found."); print(f"Details: {e}"); sys.exit(1)
# except Exception as e:
#     print(f"\nFATAL ERROR: An error occurred while loading models. Details: {e}"); sys.exit(1)

# # --- HELPER FUNCTIONS ---
# def get_feature_names_compat(enc, input_features):
#     if hasattr(enc, "get_feature_names_out"): return list(enc.get_feature_names_out(input_features));
#     names = [];
#     for feat, cats in zip(input_features, enc.categories_):
#         for c in cats: names.append(f"{feat}_{c}")
#     return names

# # def haversine(lat1, lon1, lat2, lon2):
# #     R = 6371; lat1_rad, lon1_rad, lat2_rad, lon2_rad = np.radians([lat1, lon1, lat2, lon2])
# #     d_lon = lon2_rad - lon1_rad; d_lat = lat2_rad - lat1_rad
# #     a = np.sin(d_lat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(d_lon / 2.0)**2
# #     c = 2 * np.arcsin(np.sqrt(a)); return R * c

# # predictor.py (haversine function)
# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371  # Radius of Earth in kilometers

#     # Convert to radians separately:
#     lat1_rad = np.radians(lat1)
#     lon1_rad = np.radians(lon1)
#     lat2_rad = np.radians(lat2)
#     lon2_rad = np.radians(lon2)

#     # Note: If lat1/lon1 are scalars, they remain scalars (or 0D arrays).
#     # If lat2/lon2 are Series/arrays, they remain Series/arrays, and the
#     # following NumPy operations will handle the element-wise subtraction and calculation.

#     dlat = lat2_rad - lat1_rad
#     dlon = lon2_rad - lon1_rad

#     # Haversine formula
#     a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

#     distance = R * c
#     return distance

# def prepare_input_stg1(inp: dict, pipeline: dict):
#     df = pd.DataFrame([inp])
#     df['hour_sin'] = np.sin(2 * math.pi * df['abduction_time'] / 24.0)
#     df['hour_cos'] = np.cos(2 * math.pi * df['abduction_time'] / 24.0)
#     CITY_CENTERS = {"Mumbai":(19.0761, 72.8775), "Pune":(18.5203, 73.8567), "Nagpur":(21.1497, 79.0806), "Nashik":(19.9975, 73.7898)}
    
#     # The vectorized haversine works for this one-row DataFrame calculation as well.
#     df['dist_to_nearest_city'] = df.apply(
#         lambda row: min([haversine(row['latitude'], row['longitude'], c_lat, c_lon) for c_lat, c_lon in CITY_CENTERS.values()]),
#         axis=1
#     )
    
#     X_num_scaled = pipeline['scaler'].transform(df[pipeline['num_cols']])
#     X_cat_encoded = pipeline['encoder'].transform(df[pipeline['cat_cols']])
#     cat_feature_names = get_feature_names_compat(pipeline['encoder'], pipeline['cat_cols'])
#     X_final = pd.concat([pd.DataFrame(X_num_scaled, columns=pipeline['num_cols']), pd.DataFrame(X_cat_encoded, columns=cat_feature_names)], axis=1)
#     return X_final[pipeline['X_columns']]

# # --- MAIN PREDICTION FUNCTIONS ---
# def predict_initial_case(inp: dict):
#     X_in = prepare_input_stg1(inp, PIPELINE_STG1)
#     risk_label = int(CLF_RISK.predict(X_in)[0]); risk_prob = float(CLF_RISK.predict_proba(X_in)[0].max()); rec_prob = float(CLF_RECOVERED.predict_proba(X_in)[0][1]); rec_label = 1 if rec_prob >= 0.5 else 0
#     est_recovery_time, pred_lat, pred_lon = 0.0, 0.0, 0.0
#     if rec_label == 1:
#         est_recovery_time = float(REG_TIME.predict(X_in)[0]); pred_lat = float(REG_LAT.predict(X_in)[0]); pred_lon = float(REG_LON.predict(X_in)[0])
#     return {'risk_label':risk_label,'risk_prob':risk_prob,'recovered_label':rec_label,'recovered_prob':rec_prob,'recovery_time_hours':est_recovery_time,'predicted_latitude':pred_lat,'predicted_longitude':pred_lon}

# def refine_location_with_sightings(initial_prediction: dict, sightings: list, initial_case_input: dict):
#     if not sightings or REFINEMENT_MODEL is None:
#         return initial_prediction['predicted_latitude'], initial_prediction['predicted_longitude']

#     seq_features = []
#     # Sort sightings by time, as this is critical for a trajectory model
#     for sighting in sorted(sightings, key=lambda s: s['hours_since']):
#         text_embedding = NLP_MODEL.encode(sighting['direction_text'], device=DEVICE)
#         features = [sighting['lat'], sighting['lon'], sighting['hours_since']] + list(text_embedding)
#         seq_features.append(features)
        
#     padded_seq = np.zeros((MAX_SEQ_LEN, len(seq_features[0])), dtype=np.float32)
#     seq_len = len(seq_features)
#     if seq_len > 0:
#         padded_seq[-seq_len:] = np.array(seq_features, dtype=np.float32)
    
#     seq_tensor = torch.tensor([padded_seq], dtype=torch.float32).to(DEVICE)
    
#     with torch.no_grad():
#         # The Seq2Seq model predicts the next step in the sequence
#         refined_coords = REFINEMENT_MODEL(seq_tensor, target_len=1).cpu().numpy()[0,0,:]
        
#     return float(refined_coords[0]), float(refined_coords[1])

















import os
import math
import pandas as pd
import numpy as np
import sys
from typing import Dict, Any, List, Tuple
import torch.nn as nn

# --- GLOBAL MODEL REFERENCES (Set to None initially) ---
# These will be loaded only when first accessed.
PIPELINE_STG1 = None
CLF_RISK = None
CLF_RECOVERED = None
REG_TIME = None
REG_LAT = None
REG_LON = None
REFINEMENT_MODEL = None
NLP_MODEL = None
CITY_CENTERS = {}
df = None # Raw data reference

# --- CONFIGURATION & ROBUST PATHING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR_STG1 = os.path.join(SCRIPT_DIR, "models_lgbm_tuned")
MODEL_DIR_STG2_PYTORCH = os.path.join(SCRIPT_DIR, "models_refinement_pytorch")
MAX_SEQ_LEN = 5  # Must match the training script

# NOTE: DEVICE is defined inside _load_models_if_needed
DEVICE = None 

# --- PYTORCH MODEL DEFINITIONS (These must remain outside functions) ---
# These class definitions are lightweight and must be here for Uvicorn to load.
class EncoderLSTM(nn.Module):
    # nn.Module is now imported inside the loading function
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(EncoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell

class DecoderLSTM(nn.Module):
    # nn.Module is now imported inside the loading function
    def __init__(self, output_size, hidden_size=128, num_layers=2):
        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc_out = nn.Linear(hidden_size, output_size)
    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell

class RefinementEngine(nn.Module):
    # nn.Module is now imported inside the loading function
    def __init__(self, encoder, decoder, device):
        super(RefinementEngine, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, source, target_len=1):
        batch_size = source.shape[0]
        # torch is now imported inside the loading function
        outputs = torch.zeros(batch_size, target_len, 2).to(self.device) 
        hidden, cell = self.encoder(source)
        x = source[:, -1, 0:2].unsqueeze(1)
        for t in range(target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)
            x = output
        return outputs

# --- LAZY LOADING FUNCTION (The key change) ---
def _load_models_if_needed():
    """Initializes models and global data only if they haven't been loaded yet."""
    # Move heavy imports inside this function
    try:
        import joblib
        import torch
        import torch.nn as nn
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"CRITICAL: Missing dependency for model loading: {e}", file=sys.stderr)
        sys.exit(1) # Crash hard if dependencies are missing

    global PIPELINE_STG1, CLF_RISK, CLF_RECOVERED, REG_TIME, REG_LAT, REG_LON, REFINEMENT_MODEL, NLP_MODEL, CITY_CENTERS, df, DEVICE
    
    if PIPELINE_STG1 is not None:
        return # Models are already loaded

    print("\n--- LAZY LOADING: Initializing Sachet AI Engine... ---")

    try:
        # Define device after torch is imported
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load Stage 1 (LightGBM) Models
        PIPELINE_STG1 = joblib.load(os.path.join(MODEL_DIR_STG1, 'pipeline.joblib'))
        CLF_RISK = joblib.load(os.path.join(MODEL_DIR_STG1, 'clf_risk.joblib'))
        CLF_RECOVERED = joblib.load(os.path.join(MODEL_DIR_STG1, 'clf_recovered.joblib'))
        REG_TIME = joblib.load(os.path.join(MODEL_DIR_STG1, 'reg_recovery_time.joblib'))
        REG_LAT = joblib.load(os.path.join(MODEL_DIR_STG1, 'reg_recovery_lat.joblib'))
        REG_LON = joblib.load(os.path.join(MODEL_DIR_STG1, 'reg_recovery_lon.joblib'))
        print("Stage 1 (LightGBM) Models loaded.")

        # 2. Load Stage 2 (PyTorch Trajectory) Models and NLP
        SEQ_FEATURE_SIZE = 3 + 384
        encoder_model = EncoderLSTM(SEQ_FEATURE_SIZE).to(DEVICE)
        decoder_model = DecoderLSTM(output_size=2).to(DEVICE)
        REFINEMENT_MODEL = RefinementEngine(encoder_model, decoder_model, DEVICE).to(DEVICE)
        REFINEMENT_MODEL.load_state_dict(torch.load(os.path.join(MODEL_DIR_STG2_PYTORCH, 'refinement_model.pth'), map_location=torch.device(DEVICE)))
        REFINEMENT_MODEL.eval()
        
        # Load NLP Model
        NLP_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        print("Stage 2 (PyTorch/NLP) Models loaded.")
        
        # 3. Load Global Data
        df = pd.read_csv("sachet_main_cases_2M.csv", usecols=['abduction_time','abductor_relation','region_type','recovered','recovery_latitude','recovery_longitude'])
        cities_df = pd.read_csv("worldcities.csv")
        major_cities = cities_df[cities_df['population'] > 500000]
        CITY_CENTERS = {row['city']: (row['lat'], row['lng']) for index, row in major_cities.iterrows()}
        print("--- Data Loaded Successfully. AI Engine is ready. ---")

    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: A required file was not found during lazy load: {e}", file=sys.stderr); sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR during lazy load: {e}", file=sys.stderr)
        # We must exit if the models fail to load to prevent bad predictions
        sys.exit(1)

# --- HELPER FUNCTIONS ---
def get_feature_names_compat(enc, input_features):
    # Logic remains the same
    _load_models_if_needed() # Ensure models are loaded before access
    if hasattr(enc, "get_feature_names_out"): return list(enc.get_feature_names_out(input_features));
    names = [];
    for feat, cats in zip(input_features, enc.categories_):
        for c in cats: names.append(f"{feat}_{c}")
    return names

def haversine(lat1, lon1, lat2, lon2):
    # Logic remains the same
    R = 6371; lat1_rad, lon1_rad, lat2_rad, lon2_rad = np.radians([lat1, lon1, lat2, lon2])
    d_lon = lon2_rad - lon1_rad; d_lat = lat2_rad - lat1_rad
    a = np.sin(d_lat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(d_lon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a)); return R * c

def prepare_input_stg1(inp: dict):
    # Logic remains the same
    _load_models_if_needed() 
    pipeline = PIPELINE_STG1
    
    df = pd.DataFrame([inp])
    df['hour_sin'] = np.sin(2 * math.pi * df['abduction_time'] / 24.0)
    df['hour_cos'] = np.cos(2 * math.pi * df['abduction_time'] / 24.0)
    
    # Calculate distance using the lazily loaded centers
    df['dist_to_nearest_city'] = df.apply(
        lambda row: min([haversine(row['latitude'], row['longitude'], c_lat, c_lon) for c_lat, c_lon in CITY_CENTERS.values()]),
        axis=1
    )
    
    X_num_scaled = pipeline['scaler'].transform(df[pipeline['num_cols']])
    X_cat_encoded = pipeline['encoder'].transform(df[pipeline['cat_cols']])
    cat_feature_names = get_feature_names_compat(pipeline['encoder'], pipeline['cat_cols'])
    X_final = pd.concat([pd.DataFrame(X_num_scaled, columns=pipeline['num_cols']), pd.DataFrame(X_cat_encoded, columns=cat_feature_names)], axis=1)
    return X_final[pipeline['X_columns']]

# --- MAIN PREDICTION FUNCTIONS ---
def predict_initial_case(inp: dict) -> Dict[str, Any]:
    _load_models_if_needed() # Load models on first call
    # Logic remains the same, relying on global variables CLF_RISK, etc.
    X_in = prepare_input_stg1(inp)
    risk_label = int(CLF_RISK.predict(X_in)[0]); risk_prob = float(CLF_RISK.predict_proba(X_in)[0].max()); rec_prob = float(CLF_RECOVERED.predict_proba(X_in)[0][1]); rec_label = 1 if rec_prob >= 0.5 else 0
    est_recovery_time, pred_lat, pred_lon = 0.0, 0.0, 0.0
    if rec_label == 1:
        est_recovery_time = float(REG_TIME.predict(X_in)[0]); pred_lat = float(REG_LAT.predict(X_in)[0]); pred_lon = float(REG_LON.predict(X_in)[0])
    return {'risk_label':risk_label,'risk_prob':risk_prob,'recovered_label':rec_label,'recovered_prob':rec_prob,'recovery_time_hours':est_recovery_time,'predicted_latitude':pred_lat,'predicted_longitude':pred_lon}

def refine_location_with_sightings(initial_prediction: dict, sightings: list, initial_case_input: dict) -> Tuple[float, float]:
    _load_models_if_needed() # Load models on first call
    # Logic remains the same, relying on global variables NLP_MODEL, etc.
    if not sightings or REFINEMENT_MODEL is None:
        return initial_prediction['predicted_latitude'], initial_prediction['predicted_longitude']
    
    # NLP_MODEL and torch is now available globally after _load_models_if_needed
    import torch 
    seq_features = []
    
    for sighting in sorted(sightings, key=lambda s: s['hours_since']):
        text_embedding = NLP_MODEL.encode(sighting['direction_text'], device=DEVICE) 
        features = [sighting['lat'], sighting['lon'], sighting['hours_since']] + list(text_embedding)
        seq_features.append(features)
        
    padded_seq = np.zeros((MAX_SEQ_LEN, len(seq_features[0])), dtype=np.float32)
    seq_len = len(seq_features)
    if seq_len > 0:
        padded_seq[-seq_len:] = np.array(seq_features, dtype=np.float32)
    
    seq_tensor = torch.tensor([padded_seq], dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        refined_coords = REFINEMENT_MODEL(seq_tensor, target_len=1).cpu().numpy()[0,0,:]
        
    return float(refined_coords[0]), float(refined_coords[1])
