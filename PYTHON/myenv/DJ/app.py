# import streamlit as st
# import pandas as pd
# import folium
# from streamlit_folium import st_folium
# import os
# import numpy as np
# from sklearn.cluster import KMeans
# from datetime import datetime

# # --- Mock Functions (for demonstration if predictor.py is not available) ---
# # In your real app, REMOVE THIS and ensure predictor.py is present.
# # def predict_initial_case(case_input):
# #     """MOCK FUNCTION: Replace with your actual model call."""
# #     return {
# #         'risk_label': np.random.randint(0, 3),
# #         'risk_prob': np.random.rand(),
# #         'recovered_prob': np.random.rand(),
# #         'recovery_time_hours': np.random.uniform(5, 72),
# #         'predicted_latitude': case_input['latitude'] + np.random.uniform(-0.1, 0.1),
# #         'predicted_longitude': case_input['longitude'] + np.random.uniform(-0.1, 0.1),
# #     }

# # def refine_location_with_sightings(prediction, sightings, case_input):
# #     """MOCK FUNCTION: Replace with your actual refinement model."""
# #     last_sighting = sightings[-1]
# #     return (last_sighting['lat'] + np.random.uniform(-0.05, 0.05), last_sighting['lon'] + np.random.uniform(-0.05, 0.05))

# # def haversine(lat1, lon1, lat2, lon2):
# #     """MOCK FUNCTION: Replace with your actual haversine calculation."""
# #     return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111 # Rough approximation
# # --- End of Mock Functions ---


# # --- Imports and Initial Error Checking ---
# # Uncomment the block below to use your actual predictor.py file
# try:
#     from predictor import predict_initial_case, refine_location_with_sightings, haversine
# except ImportError:
#     st.error("FATAL ERROR: The 'predictor.py' file was not found. Please ensure it is in the same directory.")
#     st.stop()
# except Exception as e:
#     st.error(f"FATAL ERROR on startup: Could not load models from predictor.py. Have you run train.py successfully? Details: {e}")
#     st.stop()

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Sachet: Predictive Alert System",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # --- Custom CSS for Styling ---
# def load_custom_css():
#     st.markdown("""
#         <style>
#             /* Main header style */
#             h1 {
#                 color: #2c3e50; /* Dark blue-grey */
#                 font-family: 'Arial', sans-serif;
#             }
#             /* Styling for metric labels */
#             .stMetric .st-ax {
#                 color: #34495e; /* Slightly lighter blue-grey */
#             }
#             /* Main container */
#             .main .block-container {
#                 padding-top: 2rem;
#             }
#             /* Primary button style */
#             div.stButton > button:first-child {
#                 background-color: #e74c3c; /* Red */
#                 color: white;
#                 border: none;
#                 border-radius: 5px;
#                 padding: 10px 24px;
#                 font-size: 16px;
#             }
#             div.stButton > button:first-child:hover {
#                 background-color: #c0392b; /* Darker red on hover */
#             }
#             /* Expander header style */
#             .streamlit-expanderHeader {
#                 background-color: #ecf0f1; /* Light grey */
#                 font-size: 1.1em;
#                 color: #2c3e50;
#                 border: 1px solid #bdc3c7;
#             }
#         </style>
#     """, unsafe_allow_html=True)

# load_custom_css()

# # --- Data Loading ---
# RANDOM_SEED = 42
# DATASET_PATH = "sachet_main_cases_2M.csv" # Your real-world historical cases file
# CITIES_DATA_PATH = "worldcities.csv"     # Your real-world cities file

# @st.cache_data
# def load_data(path, columns=None):
#     if os.path.exists(path):
#         return pd.read_csv(path, usecols=columns)
#     return pd.DataFrame() # Return empty DataFrame if file not found

# df = load_data(DATASET_PATH, columns=['abduction_time','abductor_relation','region_type','recovered','recovery_latitude','recovery_longitude'])
# cities_df = load_data(CITIES_DATA_PATH)

# if df.empty or cities_df.empty:
#     st.error(f"FATAL ERROR: A required dataset was not found. Ensure '{DATASET_PATH}' and '{CITIES_DATA_PATH}' exist.")
#     st.stop()
    
# major_cities = cities_df[cities_df['population'] > 500000]
# CITY_CENTERS = {row['city']: (row['lat'], row['lng']) for index, row in major_cities.iterrows()}

# # --- Initialize Session State ---
# if 'prediction' not in st.session_state: st.session_state.prediction = None
# if 'sightings' not in st.session_state: st.session_state.sightings = []
# if 'refined_location' not in st.session_state: st.session_state.refined_location = None
# if 'initial_case_input' not in st.session_state: st.session_state.initial_case_input = None
# if 'map_key' not in st.session_state: st.session_state.map_key = 'initial_map'

# # --- UI Layout ---
# st.title("🔔 Sachet: Missing Child Predictive Alert System")
# st.markdown("---")

# # --- Case Input Form (in main area) ---
# with st.expander("📝 Enter New Case Details to Generate Prediction", expanded=True):
#     with st.form("case_input_form"):
#         st.subheader("Child and Abduction Details")
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             age = st.slider("Child Age", 1, 18, 9, key="age")
#         with col2:
#             gender = st.selectbox("Child Gender", ["M", "F"], key="gender")
#         with col3:
#             hour = st.slider("Abduction Time (24h)", 0, 23, 17, key="hour")
#         with col4:
#             dow = st.slider("Day of Week (Mon=0)", 0, 6, 4, key="dow")

#         st.subheader("Location and Context")
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             lat = st.number_input("Last Seen Latitude", value=18.5203, format="%.4f", key="lat")
#             lon = st.number_input("Last Seen Longitude", value=73.8567, format="%.4f", key="lon")
#         with col2:
#             region_type = st.selectbox("Region Type", df['region_type'].unique(), key="region")
#             pop_density = st.number_input("Population Density", value=20000, min_value=10, key="pop")
#         with col3:
#             transport_hub = st.selectbox("Major Transport Hub Nearby?", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No', key="hub")
#             relation = st.selectbox("Abductor Relation", df['abductor_relation'].unique(), key="relation")
        
#         submitted = st.form_submit_button("Predict Initial Location Hotspot")

# if submitted:
#     case_input={'child_age':age,'child_gender':gender,'abduction_time':hour,'abductor_relation':relation,'latitude':lat,'longitude':lon,'day_of_week':dow,'region_type':region_type,'population_density':pop_density,'transport_hub_nearby':transport_hub}
#     dist=min([haversine(lat,lon,c_lat,c_lon) for c_lat,c_lon in CITY_CENTERS.values()]); case_input['dist_to_nearest_city']=dist
    
#     with st.spinner("🧠 Running initial prediction using Stage 1 AI..."):
#         st.session_state.initial_case_input=case_input
#         st.session_state.prediction=predict_initial_case(case_input)
#         st.session_state.sightings=[]
#         st.session_state.refined_location=None
#         st.session_state.start_time=datetime.now()
#         # Create a new key for the map to force a re-render
#         st.session_state.map_key = f'map_{datetime.now().timestamp()}'

# # --- Display Prediction Results ---
# if not st.session_state.prediction:
#     st.info("Enter case details above and click 'Predict' to begin analysis.")
# else:
#     pred = st.session_state.prediction
#     risk_map = {0: "Low", 1: "Medium", 2: "High"}
#     alert_map = {0: "Internal Review", 1: "Local Alert", 2: "Amber Alert"}

#     st.header("🚨 Prediction Results")
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Risk Level", f"{risk_map.get(pred['risk_label'], 'N/A')} Risk", f"Confidence: {pred['risk_prob']:.1%}")
#     col2.metric("Recommended Alert", alert_map.get(pred['risk_label'], 'N/A'))
#     col3.metric("Est. Recovery Time", f"~{int(pred['recovery_time_hours'])} hours" if pred.get('recovery_time_hours', 0) > 0 else "N/A")
#     st.markdown("---")

#     # --- Live Sightings & Map Layout ---
#     st.header("📍 Live Map and Sighting Management")
#     col_map, col_sightings = st.columns([3, 1]) # Map is 3 times wider than the sightings column

#     with col_sightings:
#         st.subheader("Add Live Sighting")
#         s_lat = st.number_input("Sighting Latitude", value=st.session_state.initial_case_input['latitude'] + 0.05, format="%.4f")
#         s_lon = st.number_input("Sighting Longitude", value=st.session_state.initial_case_input['longitude'] + 0.05, format="%.4f")
#         s_hours = st.number_input("Hours Since Abduction", min_value=0.1, step=0.5, value=5.0, format="%.1f")
#         s_text = st.text_input("Direction/Description", "e.g., heading towards highway")
        
#         if st.button("Add Sighting"):
#             st.session_state.sightings.append({'lat': s_lat, 'lon': s_lon, 'direction_text': s_text, 'hours_since': s_hours})
#             st.success("Sighting added! Map updated.")
#             st.session_state.map_key = f'map_{datetime.now().timestamp()}'
#             st.rerun()

#         if st.session_state.sightings:
#             st.subheader("Logged Sightings")
#             for i, s in enumerate(st.session_state.sightings):
#                 st.info(f"#{i+1}: Lat: {s['lat']:.3f}, Lon: {s['lon']:.3f} @ {s['hours_since']} hrs")
            
#             if st.button("Refine Prediction with Sightings"):
#                  with st.spinner("🧠 Running Stage 2 AI (Refinement)..."):
#                     r_lat, r_lon = refine_location_with_sightings(pred, st.session_state.sightings, st.session_state.initial_case_input)
#                     st.session_state.refined_location = (r_lat, r_lon)
#                     st.session_state.map_key = f'map_{datetime.now().timestamp()}'
#                     st.rerun()

#     with col_map:
#         initial_lat = st.session_state.initial_case_input['latitude']
#         initial_lon = st.session_state.initial_case_input['longitude']

#         m = folium.Map(location=[initial_lat, initial_lon], zoom_start=9, tiles="cartodbpositron")
#         folium.Marker([initial_lat, initial_lon], popup="Last Seen Location", icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)

#         # Plot Sightings
#         if st.session_state.sightings:
#             sighting_points = [(initial_lat, initial_lon)] # Start polyline from initial point
#             for s in st.session_state.sightings:
#                 folium.Marker([s['lat'], s['lon']], popup=f"Sighting @ {s['hours_since']}h", icon=folium.Icon(color='orange', icon='eye', prefix='fa')).add_to(m)
#                 sighting_points.append((s['lat'], s['lon']))
#             if len(sighting_points) > 1:
#                 folium.PolyLine(sighting_points, color="orange", weight=2.5, opacity=0.8, dash_array='5, 10').add_to(m)

#         # Plot Predicted Hotspot
#         if st.session_state.refined_location:
#             p_lat, p_lon = st.session_state.refined_location
#             p_popup, p_color, p_icon = "REFINED Hotspot", "purple", "bullseye"
#             st.success(f"AI refined primary hotspot to: {p_lat:.4f}, {p_lon:.4f}")
#         else:
#             p_lat, p_lon = pred['predicted_latitude'], pred['predicted_longitude']
#             p_popup, p_color, p_icon = "INITIAL Predicted Hotspot", "red", "star"
        
#         folium.Marker([p_lat, p_lon], popup=p_popup, icon=folium.Icon(color=p_color, icon=p_icon, prefix='fa')).add_to(m)
#         radius_m = max(500, 15000 * (1 - pred['recovered_prob']))
#         folium.Circle(radius=radius_m, location=[p_lat, p_lon], color=p_color, fill=True, fill_opacity=0.15, popup="Primary Search Radius").add_to(m)

#         st_folium(m, key=st.session_state.map_key, width='100%', height=600)


























import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime

# --- IMPORTANT: Ensure predictor.py is in the same folder ---
# --- And uncomment the following block to use your real model ---
try:
    from predictor import predict_initial_case, refine_location_with_sightings, haversine
except ImportError:
    st.error("FATAL ERROR: The 'predictor.py' file was not found. Please ensure it is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"FATAL ERROR on startup: Could not load models from predictor.py. Have you run train.py successfully? Details: {e}")
    st.stop()


# --- Mock Functions (for standalone testing if predictor.py is missing) ---
# --- DELETE or comment out this block if you are using your real predictor.py ---
# def predict_initial_case(case_input):
#     return { 'risk_label': np.random.randint(0, 3), 'risk_prob': np.random.rand(), 'recovered_prob': np.random.rand(), 'recovery_time_hours': np.random.uniform(5, 72), 'predicted_latitude': case_input['latitude'] + np.random.uniform(-0.1, 0.1), 'predicted_longitude': case_input['longitude'] + np.random.uniform(-0.1, 0.1) }
# def refine_location_with_sightings(prediction, sightings, case_input):
#     last_sighting = sightings[-1]; return (last_sighting['lat'] + np.random.uniform(-0.05, 0.05), last_sighting['lon'] + np.random.uniform(-0.05, 0.05))
# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371; lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2]); dlon = lon2 - lon1; dlat = lat2 - lat1
#     a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2; c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
#     return R * c
# --- End of Mock Functions Block ---


# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="Sachet: Predictive Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def apply_custom_styling():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');
            body { font-family: 'Roboto', sans-serif; }
            .st-emotion-cache-16txtl3 { background-color: #262730; border: 1px solid #333; border-radius: 10px; }
            div.stButton > button[kind="primary"] { background-color: #e74c3c; color: white; border: none; border-radius: 5px; padding: 12px 24px; font-size: 16px; font-weight: 500; width: 100%; transition: background-color 0.3s ease; }
            div.stButton > button[kind="primary"]:hover { background-color: #c0392b; }
        </style>
    """, unsafe_allow_html=True)

apply_custom_styling()

# --- Data Loading ---
RANDOM_SEED = 42
DATASET_PATH = "sachet_main_cases_2M.csv"
CITIES_DATA_PATH = "worldcities.csv"

@st.cache_data
def load_data(path, columns=None):
    if os.path.exists(path): return pd.read_csv(path, usecols=columns)
    return pd.DataFrame({'region_type': ['Urban'], 'abductor_relation': ['Stranger']})

df = load_data(DATASET_PATH, columns=['abduction_time','abductor_relation','region_type','recovered','recovery_latitude','recovery_longitude'])
cities_df = load_data(CITIES_DATA_PATH)
major_cities = cities_df[cities_df['population'] > 500000] if 'population' in cities_df else pd.DataFrame()
CITY_CENTERS = {row['city']: (row['lat'], row['lng']) for index, row in major_cities.iterrows()}

# --- Initialize Session State ---
for key in ['prediction', 'sightings', 'refined_location', 'initial_case_input']:
    if key not in st.session_state: st.session_state[key] = None
if 'map_key' not in st.session_state: st.session_state.map_key = 'initial_map'

# --- App Layout (Two Columns) ---
col_inputs, col_outputs = st.columns([2, 3])

with col_inputs:
    st.header("Case Controls")
    with st.container(border=True):
        st.subheader("Child & Abduction Info")
        age = st.slider("Child Age", 1, 18, 9); gender = st.selectbox("Child Gender", ["M", "F"])
        hour = st.slider("Abduction Time (24h)", 0, 23, 18); dow = st.slider("Day of Week (Mon=0)", 0, 6, 4)

        st.subheader("Location & Context")
        lat = st.number_input("Last Seen Latitude", value=18.5203, format="%.4f"); lon = st.number_input("Last Seen Longitude", value=73.8567, format="%.4f")
        region_type = st.selectbox("Region Type", df['region_type'].unique()); pop_density = st.number_input("Population Density", value=8000000)
        
        st.subheader("Abductor Info")
        transport_hub = st.selectbox("Major Transport Hub Nearby?", ["Yes", "No"]); relation = st.selectbox("Abductor Relation", df['abductor_relation'].unique())

        if st.button("Predict Location Hotspots", type="primary"):
            case_input={'child_age':age, 'child_gender':gender, 'abduction_time':hour, 'abductor_relation':relation, 'latitude':lat, 'longitude':lon, 'day_of_week':dow, 'region_type':region_type, 'population_density':pop_density, 'transport_hub_nearby': 1 if transport_hub == 'Yes' else 0}
            dist = min([haversine(lat, lon, c_lat, c_lon) for c_lat, c_lon in CITY_CENTERS.values()]) if CITY_CENTERS else 0
            case_input['dist_to_nearest_city'] = dist
            with st.spinner("🧠 Running Initial Prediction..."):
                st.session_state.initial_case_input = case_input; st.session_state.prediction = predict_initial_case(case_input)
                st.session_state.sightings = []; st.session_state.refined_location = None; st.session_state.map_key = f'map_{datetime.now().timestamp()}'

with col_outputs:
    st.header("Prediction & Live Analysis")
    if not st.session_state.prediction:
        st.info("Enter case details on the left and click 'Predict' to begin.")
    else:
        pred = st.session_state.prediction
        risk_map = {0: "Low", 1: "Medium", 2: "High"}; color_map = {0:"green", 1:"orange", 2:"red"}
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Level", f"{risk_map.get(pred['risk_label'], 'N/A')}", f"Confidence: {pred['risk_prob']:.1%}")
        m2.metric("Probability of Recovery", f"{pred['recovered_prob']:.1%}")
        m3.metric("Est. Recovery Time", f"~{int(pred['recovery_time_hours'])} hours")
        st.markdown("---", unsafe_allow_html=True)

        # --- MAP VISUALIZATION ---
        m = folium.Map(location=[lat, lon], zoom_start=8, tiles="OpenStreetMap")
        folium.Marker([lat,lon],popup="Last Seen",icon=folium.Icon(color="blue",icon="info-sign")).add_to(m)
        if st.session_state.sightings:
            sighting_points = [(lat, lon)]
            for s in st.session_state.sightings:
                folium.Marker([s['lat'], s['lon']], popup=f"Sighting @ {s['hours_since']}h: {s['direction_text']}", icon=folium.Icon(color='orange', icon='eye', prefix='fa')).add_to(m)
                sighting_points.append((s['lat'], s['lon']))
            folium.PolyLine(sighting_points, color="orange", weight=2.5, opacity=0.8, dash_array='5, 10').add_to(m)

        p_lat, p_lon = (st.session_state.refined_location or (pred['predicted_latitude'], pred['predicted_longitude']))
        if st.session_state.refined_location: folium.Marker([p_lat, p_lon], popup="REFINED Hotspot", icon=folium.Icon(color="purple", icon="bullseye", prefix='fa')).add_to(m)
        else: folium.Marker([p_lat, p_lon], popup="INITIAL Hotspot", icon=folium.Icon(color="red", icon="star", prefix='fa')).add_to(m)
        radius_m = max(500, 15000 * (1-pred['recovered_prob']));folium.Circle(radius=radius_m, location=[p_lat, p_lon], color="purple" if st.session_state.refined_location else "red", fill=True, fill_opacity=0.15).add_to(m)

        recovered_df=df[df['recovered']==1].dropna(subset=['recovery_latitude','recovery_longitude']) if 'recovered' in df else pd.DataFrame()
        if not recovered_df.empty:
            time_mask=(recovered_df['abduction_time'].between(hour-2,hour+2)); context_mask=(recovered_df['abductor_relation']==relation)&(recovered_df['region_type']==region_type)
            similar_cases=recovered_df[time_mask & context_mask].copy()
            if len(similar_cases) >= 10:
                distances=haversine(p_lat, p_lon, similar_cases['recovery_latitude'], similar_cases['recovery_longitude']); relevant_cases = similar_cases[distances < 150]
                if len(relevant_cases) >= 10:
                    coords=relevant_cases[['recovery_latitude','recovery_longitude']].values; k=max(1, min(4, len(relevant_cases)//15)); kmeans=KMeans(n_clusters=k, n_init='auto', random_state=RANDOM_SEED).fit(coords)
                    for i in range(k):
                        center_lat,center_lon=kmeans.cluster_centers_[i]; folium.Circle(radius=10000,location=[center_lat,center_lon],color=color_map.get(pred['risk_label']), fill=True, fill_opacity=0.1, popup=f"Historical Hotspot {i+1}").add_to(m)
        st_folium(m, key=st.session_state.map_key, width='100%', height=500)
        if st.session_state.refined_location: st.success(f"AI refined primary hotspot to: {st.session_state.refined_location[0]:.4f}, {st.session_state.refined_location[1]:.4f}")

        # --- LOG SIGHTING FORM (NO EXPANDER) ---
        with st.container(border=True):
            st.subheader("🕵️ Log a Sighting to Refine Prediction")
            s_lat = st.number_input("Sighting Latitude", value=lat + 0.05, format="%.4f", key="s_lat")
            s_lon = st.number_input("Sighting Longitude", value=lon + 0.05, format="%.4f", key="s_lon")
            s_hours = st.number_input("Hours Since Abduction", min_value=0.1, value=16.0, format="%.1f", key="s_hours")
            s_text = st.text_input("Direction Description (e.g., 'moving towards Solapur on NH 65')", key="s_text")
            if st.button("Add Sighting and Refine"):
                st.session_state.sightings.append({'lat': s_lat, 'lon': s_lon, 'hours_since': s_hours, 'direction_text': s_text})
                with st.spinner("🧠 Re-running model with new sighting..."):
                    r_lat, r_lon = refine_location_with_sightings(pred, st.session_state.sightings, st.session_state.initial_case_input)
                    st.session_state.refined_location = (r_lat, r_lon)
                    st.session_state.map_key = f'map_refined_{datetime.now().timestamp()}'
                st.rerun()