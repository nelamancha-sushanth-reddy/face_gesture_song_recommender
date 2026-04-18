import os
import warnings
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
warnings.filterwarnings('ignore', category=UserWarning)  

import time
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

import logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

if not os.path.exists('uploads'):
    os.makedirs('uploads')

CSV_PATH = "spotify_tracks.csv"
MODEL_PATH = "model.h5"

LANGUAGE_MAP = {
    "hindi": "hindi",
    "english": "english",
    "tamil": "tamil",
    "telugu": "telugu"
}
 
# Emotion to Valence Range Mapping
# Valence: 0 = low energy/sad, 0.5 = neutral, 1 = high energy/happy
EMOTION_VALENCE_MAP = {
    "angry": (0.4, 0.8),      # High energy, intense
    "disgusted": (0.1, 0.4),  # Low energy, negative
    "fearful": (0.1, 0.4),    # Map to disgusted range (low energy, dark)
    "happy": (0.7, 1.0),      # High energy, positive
    "neutral": (0.35, 0.65),  # Medium energy, balanced
    "sad": (0.0, 0.35),       # Low energy, melancholic
    "surprised": (0.6, 0.9)   # High energy, uplifting
}

print("Loading and processing dataset...")
try:
    df = pd.read_csv(CSV_PATH)
    df['name'] = df['track_name']
    df['artist'] = df['artist_name']
    df['link'] = df['track_url']
    df['language'] = df['language'].str.lower().str.strip()
    
    # Ensure valence is numeric
    df['valence'] = pd.to_numeric(df['valence'], errors='coerce')
    df = df.dropna(subset=['valence'])
    
    df = df[['name', 'artist', 'link', 'language', 'valence']]
    print("\n" + "---" * 10)
    print("VERIFICATION STEP: Found these language codes in CSV:", df['language'].unique())
    print(f"Total songs loaded: {len(df)}")
    print("Valence range:", f"{df['valence'].min():.2f} to {df['valence'].max():.2f}")
    print("---" * 10 + "\n")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"FATAL ERROR: The CSV file was not found at the path: {CSV_PATH}")
    exit()
except KeyError as e:
    print(f"FATAL ERROR: A required column is missing from the CSV: {e}")
    exit()


def get_recommendations(emotion_list, language_code="hi"):
    """
    Recommendation system using valence-based emotion-to-music matching
    Returns songs that match the detected emotion and language
    """
    print("\n" + "---" * 10)
    print("MUSIC RECOMMENDATION SYSTEM")
    print(f"--> Received Emotion: {emotion_list[0] if emotion_list else 'None'}")
    print(f"--> Received Language Code: '{language_code}'")
    if not emotion_list: return pd.DataFrame(columns=["name", "artist", "link"])
    emotion = emotion_list[0]
    lang_df = df[df['language'] == language_code]
    print(f"--> STEP 1: Found {len(lang_df)} songs for language '{language_code}'.")
    if lang_df.empty: return pd.DataFrame(columns=["name", "artist", "link"])
    n_tracks, band_size = len(lang_df), len(lang_df) // 5
    print(f"--> STEP 2: Slicing for emotion '{emotion}'. Num songs: {n_tracks}, Band size: {band_size}")
    emotion_df = pd.DataFrame()
    
    # Get valence range for this emotion
    primary_emotion = emotion.lower()
    valence_range = EMOTION_VALENCE_MAP.get(primary_emotion, (0.35, 0.65))
    
    # Filter songs within the valence range
    emotion_df = lang_df[
        (lang_df['valence'] >= valence_range[0]) & 
        (lang_df['valence'] <= valence_range[1])
    ].copy()
    
    # If not enough songs, expand range
    if len(emotion_df) < 10:
        expanded_range = (
            max(0, valence_range[0] - 0.2),
            min(1, valence_range[1] + 0.2)
        )
        emotion_df = lang_df[
            (lang_df['valence'] >= expanded_range[0]) & 
            (lang_df['valence'] <= expanded_range[1])
        ].copy()
    
    if emotion_df.empty:
        emotion_df = lang_df
    
    print(f"   - Found {len(emotion_df)} songs in the emotional slice.")
    num_to_sample = min(30, len(emotion_df))
    if num_to_sample == 0: return pd.DataFrame(columns=["name", "artist", "link"])
    
    # Prefer songs closest to target valence
    target_valence = (valence_range[0] + valence_range[1]) / 2
    emotion_df['valence_distance'] = abs(emotion_df['valence'] - target_valence)
    recommended_songs = emotion_df.nsmallest(num_to_sample, 'valence_distance')
    recommended_songs = recommended_songs.drop('valence_distance', axis=1)
    
    print(f"--> STEP 3: Successfully sampled {len(recommended_songs)} songs.")
    print("---" * 10 + "\n")
    
    return recommended_songs


def process_emotions(emotion_list, confidence_list=None):
    """
    Improved emotion processing with confidence weighting
    Returns the most confident emotion
    """
    if not emotion_list:
        return None
    
    # If we have one detection, return it
    if len(emotion_list) == 1:
        return emotion_list[0]
    
    # If we have multiple detections with confidence scores, use weighted selection
    if confidence_list and len(confidence_list) == len(emotion_list):
        weighted_emotions = {}
        for emotion, confidence in zip(emotion_list, confidence_list):
            weighted_emotions[emotion] = weighted_emotions.get(emotion, 0) + confidence
        most_confident_emotion = max(weighted_emotions, key=weighted_emotions.get)
    else:
        # Otherwise, pick the most common emotion
        emotion_counts = Counter(emotion_list)
        most_confident_emotion = emotion_counts.most_common(1)[0][0]
    
    print(f"Detected emotions from faces: {emotion_list}")
    print(f"Selected emotion: {most_confident_emotion}")
    
    return most_confident_emotion

print("Loading emotion detection model...")
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.load_weights(MODEL_PATH)
print("Model loaded successfully.")
print("\n" + "="*60)
print("APP IS READY!")
print("LOGGING SYSTEM IS ACTIVE")
print("="*60)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/save_image", methods=["POST"])
def save_image():
    timestamp = int(time.time())
    upload_dir = os.path.join('uploads', f'image_{timestamp}')
    os.makedirs(upload_dir, exist_ok=True)
    image_path = os.path.join(upload_dir, 'uploaded.jpg')
    with open(image_path, "wb") as f:
        f.write(request.data)

    print(f"\n{'='*50}")
    print(f"Processing image: {image_path}")
    print(f"Image size: {len(request.data)} bytes")
    print(f"{'='*50}")

    img = cv2.imread(image_path)
    
    if img is None:
        print("ERROR: Could not read image - file might be corrupted or invalid format")
        return jsonify({
            "message": "Could not read the uploaded image. It might be corrupted.",
            "emotion_found": None,
            "links": []
        }), 400

    print(f"Image shape: {img.shape}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Gray image shape: {gray.shape}")
    
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    print(f"Face cascade loaded: {face_cascade_path}")
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(f"Number of faces detected: {len(faces)}")

    if len(faces) == 0:
        print("WARNING: No faces detected in image")
        return jsonify({
            "message": "No faces detected in the image. Please try again with a clear face.",
            "emotion_found": None,
            "links": []
        })

    emotion_list = []
    confidence_list = []
    
    for idx, (x, y, w, h) in enumerate(faces):
        print(f"\nProcessing face {idx+1}/{len(faces)}: position=({x},{y}), size=({w},{h})")
        roi_gray = gray[y:y+h, x:x+w]
        print(f"  ROI shape: {roi_gray.shape}")
        
        resized_img = cv2.resize(roi_gray, (48, 48))
        expanded_img = np.expand_dims(np.expand_dims(resized_img, axis=-1), axis=0)
        print(f"  Resized/expanded shape: {expanded_img.shape}")
        
        predictions = model.predict(expanded_img, verbose=0)
        max_index = int(np.argmax(predictions))
        confidence = float(predictions[0][max_index])
        
        detected_emotion_name = emotion_dict[max_index]
        emotion_list.append(detected_emotion_name)
        confidence_list.append(confidence)
        
        print(f"  Detected: {detected_emotion_name}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  All predictions: {predictions[0]}")

    print(f"\nTotal emotions detected: {len(emotion_list)}")
    print(f"Emotion list: {emotion_list}")
    print(f"Confidence list: {confidence_list}")
    
    # Get the best emotion based on confidence
    detected_emotion = process_emotions(emotion_list, confidence_list)
    print(f"Final selected emotion: {detected_emotion}")
    
    if not detected_emotion:
        print("ERROR: process_emotions returned None")
        return jsonify({
            "message": "Could not process emotions. Please try again.",
            "emotion_found": None,
            "links": []
        })

    language_name = request.headers.get("X-Language", "hindi").lower()
    language = LANGUAGE_MAP.get(language_name, "hindi")
    print(f"Language: {language}")
    
    print(f"CALLING get_recommendations with emotion={detected_emotion}, language={language}")
    rec_data = get_recommendations([detected_emotion], language)
    print(f"Recommended songs: {len(rec_data)}")
    
    results = []
    
    for _, row in rec_data.iterrows():
        results.append({"song": row['name'], "artist": row['artist'], "link": row['link']})

    print(f"{'='*50}\n")
    
    return jsonify({
        "message": f"Detected emotion: {detected_emotion}",
        "emotion_found": detected_emotion,
        "links": results
    })


@app.route("/change_language", methods=["POST"])
def change_language():
    data = request.get_json()
    detected_emotion = data.get("emotion")
    language_name = data.get("language", "hindi").lower()
    language = LANGUAGE_MAP.get(language_name, "hindi")
    
    if not detected_emotion:
        return jsonify({"links": [], "message": "Emotion not provided."})
        
    rec_data = get_recommendations([detected_emotion], language)
    results = []
    
    for _, row in rec_data.iterrows():
        results.append({"song": row['name'], "artist": row['artist'], "link": row['link']})
        
    return jsonify({"links": results})

 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)
    app.run(debug=False, host='localhost', port=5000)
