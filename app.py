from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from PIL import Image
import io
import base64
from model import EmotionRec

app = Flask(__name__)
app.secret_key = os.urandom(24)  # for flash messages

# Load your emotion recognition model
model = EmotionRec()
model.load_state_dict(torch.load('emotion_recognition_model.pth'))
model.eval()

# Load your song dataset
song_features_df = pd.read_csv("genres_v2.csv", low_memory=False)

# Spotify setup
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="420b73e02a0d4ec6aa5fec7ac8ae6b64",
    client_secret="3522e9079ba9479da0b22ce748c61d22",
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-read-playback-state user-modify-playback-state user-read-currently-playing"
))

# Other global variables and functions
emotion_dict = {0: "Angry", 1: "Happy", 2: "Sad", 3: "Calm"}
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def emotion_recog(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert image to grayscale only if it's not already
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detect faces
    facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        return None
    
    # Process the first face found
    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y + h, x:x + w]
    
    # Preprocess the face image
    cropped_img = cv2.resize(roi_gray, (48, 48))
    cropped_img = np.array(cropped_img, dtype=np.uint8)
    pil_img = Image.fromarray(cropped_img)
    cropped_img = transform(pil_img).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        prediction = model(cropped_img)
        maxindex = int(torch.argmax(prediction))
    
    return emotion_dict[maxindex]

def get_random_songs(mood, n=5):
    mood_songs = song_features_df[song_features_df['mood'] == mood]
    return mood_songs[['song_name', 'uri']].sample(n).to_dict('records')

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Select relevant features for PCA (you might adjust this based on your analysis)
X = song_features_df[['valence', 'energy', 'danceability', 'loudness', 'acousticness']]

# Apply PCA to reduce to 4 principal components
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)

# Perform K-Means clustering into 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Add clusters and mood mapping to the dataframe
song_features_df['cluster'] = clusters

# Map clusters to moods
cluster_to_mood = {
    0: 'angry',   # Low valence, high energy
    1: 'sad',     # Very low valence, low energy
    2: 'calm',    # Low valence, low energy
    3: 'happy'    # High valence, high energy
}
song_features_df['mood'] = song_features_df['cluster'].map(cluster_to_mood)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image uploaded')
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            flash('No image selected')
            return redirect(request.url)
        
        if file:
            # Read the image file
            image_data = file.read()
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)
            
            # Recognize emotion
            emotion = emotion_recog(image_np)
            
            if emotion is None:
                flash('No face detected in the image')
                return redirect(request.url)
            
            # Map emotion to mood
            mood_map = {"Angry": "angry", "Happy": "happy", "Sad": "sad", "Calm": "calm"}
            mood = mood_map.get(emotion, "happy")
            
            # Get song recommendations
            recommended_songs = get_random_songs(mood)
            
            return render_template('results.html', emotion=emotion, mood=mood, recommendations=recommended_songs)
    
    return render_template('index.html')

@app.route('/play/<uri>')
def play_song(uri):
    try:
        sp.start_playback(uris=[uri])
        flash('Song started playing!')
    except Exception as e:
        flash(f'Error playing song: {str(e)}')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)