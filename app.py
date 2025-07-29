# =============================================================================
# 1. IMPORTS
# =============================================================================
import streamlit as st
import librosa
import numpy as np
import joblib
import os

# =============================================================================
# 2. PAGE CONFIGURATION AND STYLING
# =============================================================================
# Set the page configuration for a more professional look
st.set_page_config(
    page_title="VibeCheck | Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to inject custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Define some basic styles directly.
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
    }
    /* Title styling */
    h1 {
        color: #1DB954; /* Spotify Green */
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
    }
    /* Button styling */
    .stButton>button {
        color: #FFFFFF;
        background-color: #1DB954;
        border-radius: 20px;
        border: 1px solid #1DB954;
        padding: 10px 24px;
        font-weight: bold;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #040404;
    }
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #1DB954;
        background-color: #282828;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 3. LOAD MODEL AND ASSETS
# =============================================================================
# Use caching to load the model only once, improving performance.
@st.cache_resource
def load_assets():
    """Loads the pre-trained model, scaler, and encoder."""
    model = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('genre_encoder.pkl')
    return model, scaler, encoder

try:
    model, scaler, encoder = load_assets()
except FileNotFoundError:
    st.error("Model files not found. Please run the `train_model.py` script first.")
    st.stop()


# =============================================================================
# 4. FEATURE EXTRACTION FUNCTION (CORRECTED VERSION)
# =============================================================================
def extract_features(file_path):
    """
    Extracts audio features and includes debugging prints.
    """
    try:
        y, sr = librosa.load(file_path, mono=True, duration=30)
        
        # --- Print to check inputs ---
        print(f"--- Starting Feature Extraction for {file_path} ---")
        print(f"Audio loaded. Sample rate: {sr}, Duration: {len(y)/sr:.2f}s")

        # --- Extract features ---
        chroma_stft_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        chroma_stft_var = np.var(librosa.feature.chroma_stft(y=y, sr=sr))
        
        rms_mean = np.mean(librosa.feature.rms(y=y))
        rms_var = np.var(librosa.feature.rms(y=y))
        
        spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_centroid_var = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        spectral_bandwidth_mean = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_bandwidth_var = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        rolloff_var = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        zero_crossing_rate_mean = np.mean(librosa.feature.zero_crossing_rate(y))
        zero_crossing_rate_var = np.var(librosa.feature.zero_crossing_rate(y))
        
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmony_mean = np.mean(y_harmonic)
        harmony_var = np.var(y_harmonic)
        perceptr_mean = np.mean(y_percussive)
        perceptr_var = np.var(y_percussive)
        
        # --- IMPORTANT DEBUGGING STEP FOR TEMPO ---
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = tempo[0] # <--- ADD THIS LINE TO EXTRACT THE NUMBER
        print(f"DEBUG: Type of 'tempo' variable is: {type(tempo)}")
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_mean = mfccs.mean(axis=1)
        mfccs_var = mfccs.var(axis=1)
        print(f"DEBUG: Shape of 'mfccs_mean' is: {mfccs_mean.shape}")
        
        # --- Combine features ---
        features = [
            chroma_stft_mean, chroma_stft_var, rms_mean, rms_var, 
            spectral_centroid_mean, spectral_centroid_var, spectral_bandwidth_mean, 
            spectral_bandwidth_var, rolloff_mean, rolloff_var, 
            zero_crossing_rate_mean, zero_crossing_rate_var, harmony_mean, 
            harmony_var, perceptr_mean, perceptr_var, tempo
        ]
        
        features.extend(mfccs_mean)
        features.extend(mfccs_var)
        
        print(f"DEBUG: Total number of features collected: {len(features)}")

        # --- Final conversion ---
        try:
            final_features = np.array(features).reshape(1, -1)
            print("--- Feature extraction successful ---")
            return final_features
        except ValueError as e:
            print("\n--- ERROR OCCURRED DURING FINAL CONVERSION ---")
            print("This usually means one of the features is not a single number.")
            # Print each feature's type to find the culprit
            for i, feat in enumerate(features):
                print(f"Feature {i}: value={feat}, type={type(feat)}")
            raise e

    except Exception as e:
        st.error(f"An unexpected error occurred in extract_features: {e}")
        return None

# =============================================================================
# 5. USER INTERFACE
# =============================================================================

# --- HEADER ---
st.title("VibeCheck 🎧")
st.markdown("### Upload a song and let our AI tell you its genre!")

# --- SIDEBAR ---
with st.sidebar:
    st.header("About VibeCheck")
    st.write("""
    This app uses a K-Nearest Neighbors (KNN) model to classify the genre of a song based on its audio features.
    It was trained on the **GTZAN Dataset**.
    """)
    st.subheader("How it works for recommendations:")
    st.write("""
    Music services analyze these features to understand song similarity. If you like a song with a certain 'vibe', they can recommend others with a similar vibe.
    """)
    st.subheader("Genres in the dataset:")
    st.write(list(encoder.classes_))

# --- MAIN CONTENT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Your Audio File")
    uploaded_file = st.file_uploader(
        "Choose a .wav or .mp3 file",
        type=["wav", "mp3"]
    )

if uploaded_file is not None:
    with open(os.path.join("temp_audio_file.wav"), "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with col1:
        st.audio(uploaded_file, format='audio/wav')

    with col2:
        st.subheader("Prediction")
        if st.button("Classify Genre"):
            with st.spinner("Analyzing the vibes... Please wait."):
                audio_features = extract_features("temp_audio_file.wav")
                
                if audio_features is not None:
                    scaled_features = scaler.transform(audio_features)
                    prediction_idx = model.predict(scaled_features)
                    predicted_genre = encoder.inverse_transform(prediction_idx)[0]
                    
                    st.success(f"We're feeling a **{predicted_genre.upper()}** vibe from this track!")
                    st.balloons()