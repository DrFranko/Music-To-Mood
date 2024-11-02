from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import torch
from torchvision import transforms
import pandas as pd
import requests
from PIL import Image
import io
import os
from model import EmotionRec

app = Flask(__name__)
app.secret_key = os.urandom(24)

model = EmotionRec()
model.load_state_dict(torch.load('emotion_recognition_model.pth', weights_only=True))  
model.eval()

song_features_df = pd.read_csv("genres_v2.csv", low_memory=False)

JAMENDO_CLIENT_ID = "bfa2f12a" 
JAMENDO_API_URL = "https://api.jamendo.com/v3.0/tracks/"

emotion_dict = {0: "Angry", 1: "Happy", 2: "Sad", 3: "Calm"}
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

X = song_features_df[['valence', 'energy', 'danceability', 'loudness', 'acousticness']]
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_pca)
song_features_df['cluster'] = clusters

cluster_to_mood = {
    0: 'angry',
    1: 'sad',
    2: 'calm',
    3: 'happy'
}
song_features_df['mood'] = song_features_df['cluster'].map(cluster_to_mood)

def emotion_recog(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        return None
    
    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y + h, x:x + w]
    cropped_img = cv2.resize(roi_gray, (48, 48))
    cropped_img = np.array(cropped_img, dtype=np.uint8)
    pil_img = Image.fromarray(cropped_img)
    cropped_img = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        prediction = model(cropped_img)
        maxindex = int(torch.argmax(prediction))
    
    return emotion_dict[maxindex]


def get_random_songs(mood, n=5):
    mood_to_tag = {
        'angry': 'rock',
        'happy': 'pop',
        'sad': 'acoustic',
        'calm': 'chillout'
    }
    tag = mood_to_tag.get(mood, 'pop')
    
    params = {
        'client_id': JAMENDO_CLIENT_ID,
        'format': 'json',
        'limit': n,
        'tags': tag,
        'order': 'popularity_total'
    }
    
    try:
        response = requests.get(JAMENDO_API_URL, params=params)
        if response.status_code == 200:
            tracks = response.json().get('results', [])
            return [{'song_name': track['name'], 'audio_url': track['audio']} for track in tracks]
        else:
            flash("Error fetching songs from Jamendo API")
            return []
    except Exception as e:
        flash(f"API Error: {str(e)}")
        return []


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
            image_data = file.read()
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)
            
            
            emotion = emotion_recog(image_np)
            if emotion is None:
                flash('No face detected in the image')
                return redirect(request.url)

            mood_map = {"Angry": "angry", "Happy": "happy", "Sad": "sad", "Calm": "calm"}
            mood = mood_map.get(emotion, "happy")

            
            recommended_songs = get_random_songs(mood)

            return render_template('results.html', emotion=emotion, mood=mood, recommendations=recommended_songs)
    
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
