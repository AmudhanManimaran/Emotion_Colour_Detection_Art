from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import cv2 # type: ignore
import numpy as np
from sklearn.cluster import KMeans # type: ignore

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simple color to emotion mapping
color_emotion_map = {
    'red': 'anger',
    'blue': 'sadness',
    'yellow': 'happiness',
    'green': 'calmness',
    'black': 'fear',
    'white': 'neutral'
}

def get_dominant_color(image_path, k=3):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))  # Resize for faster processing
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img)
    colors = kmeans.cluster_centers_.astype(int)

    return colors[0]  # Return the dominant color

def classify_emotion_from_color(rgb):
    r, g, b = rgb
    if r > 150 and g < 100 and b < 100:
        return 'anger'
    elif b > 150:
        return 'sadness'
    elif r > 200 and g > 200 and b < 100:
        return 'happiness'
    elif g > 150:
        return 'calmness'
    elif r < 50 and g < 50 and b < 50:
        return 'fear'
    elif r > 200 and g > 200 and b > 200:
        return 'neutral'
    else:
        return 'unknown'

@app.route('/')
def gallery():
    df = pd.read_csv(os.path.join('data', 'labels.csv'))
    return render_template('gallery.html', images=df.to_dict(orient='records'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            dominant_color = get_dominant_color(filepath)
            emotion = classify_emotion_from_color(dominant_color)

            return render_template('upload.html', uploaded_image=file.filename, emotion=emotion)

    return render_template('upload.html', uploaded_image=None, emotion=None)

if __name__ == '__main__':
    app.run(debug=True)
