# Emotion-Based Music Recommendation System
This project is a Flask web application that detects human emotions from images and recommends music based on the detected emotion. The emotion recognition is powered by a PyTorch deep learning model, while music recommendations are fetched from the Jamendo API.

## Features
### Emotion Detection: 
Upload an image, and the system will detect emotions such as Angry, Happy, Sad, or Calm.
### Music Recommendation: 
Based on the detected emotion, the system recommends music tracks from Jamendo that match the user's mood.
### Flask Web Application:
A user-friendly web interface to upload images and view recommendations.
## Technologies Used
### Backend: 
Flask, PyTorch
### Frontend: 
HTML, CSS, Jinja2 templates
### Music API: 
Jamendo API
### Machine Learning: 
Convolutional Neural Networks (CNN) for emotion recognition
### Data Processing: 
NumPy, Pandas
### Music Dataset: 
genres_v2.csv for song feature clustering
### Model: 
Pretrained emotion recognition model EmotionRec

## Installation
### Clone the repository:
git clone https://github.com/your-username/emotion-music-recommendation.git

### Navigate to the project directory:
cd emotion-music-recommendation

### Install the required dependencies:
pip install -r requirements.txt

### Download the pretrained emotion recognition model and place it in the project root:
emotion_recognition_model.pth

### Run the Flask application:
python app.py
