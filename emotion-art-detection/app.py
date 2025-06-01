import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from collections import Counter
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # type: ignore

# Flask app configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load color-to-emotion mapping
LABELS_PATH = 'data/labels.csv'
color_emotion_df = pd.read_csv(LABELS_PATH)

# Utility Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_dominant_colors(image_path, n_colors=5):
    """Extract dominant colors from the uploaded image using KMeans clustering."""
    image = Image.open(image_path).convert('RGB').resize((150, 150))
    pixels = np.array(image).reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    color_counts = Counter(labels)
    sorted_indices = [idx for idx, _ in sorted(color_counts.items(), key=lambda x: x[1], reverse=True)]
    dominant_colors = [centers[i] for i in sorted_indices]

    return dominant_colors, color_counts

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def map_color_to_emotion(rgb_color):
    """Find the closest color in the CSV and return the associated emotion."""
    def euclidean_dist(c1, c2):
        return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

    all_colors = color_emotion_df[['R', 'G', 'B']].values
    emotions = color_emotion_df['Emotion'].values

    distances = [euclidean_dist(rgb_color, tuple(c)) for c in all_colors]
    closest_index = distances.index(min(distances))
    return emotions[closest_index]

def plot_color_chart(dominant_colors, color_counts):
    """Plot and save a bar chart of dominant colors."""
    total = sum(color_counts.values())
    hex_colors = [rgb_to_hex(color) for color in dominant_colors]
    values = [color_counts[i] for i in range(len(dominant_colors))]

    plt.figure(figsize=(6, 2))
    plt.bar(range(len(dominant_colors)), values, color=hex_colors)
    plt.xticks(range(len(hex_colors)), hex_colors, rotation=45, fontsize=10)
    plt.title('Dominant Colors')
    plt.tight_layout()

    chart_path = os.path.join(app.config['UPLOAD_FOLDER'], 'color_chart.png')
    plt.savefig(chart_path)
    plt.close()

    return 'color_chart.png'

def plot_emotion_chart(emotion_counts):
    """Plot and save a bar chart of detected emotions."""
    plt.figure(figsize=(6, 4))
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())

    plt.bar(emotions, counts, color='skyblue')
    plt.title('Emotion Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()

    chart_path = os.path.join(app.config['UPLOAD_FOLDER'], 'emotion_chart.png')
    plt.savefig(chart_path)
    plt.close()

    return 'emotion_chart.png'

# Routes
@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)

    # Save uploaded image
    filename = secure_filename('uploaded_image.jpg')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Analyze image
    dominant_colors, color_counts = get_dominant_colors(filepath)
    mapped_emotions = [map_color_to_emotion(color) for color in dominant_colors]
    emotion_counts = Counter(mapped_emotions)

    # Generate charts
    color_chart = plot_color_chart(dominant_colors, color_counts)
    emotion_chart = plot_emotion_chart(emotion_counts)

    return render_template('result.html',
                           uploaded_image=filename,
                           color_chart=color_chart,
                           emotion_chart=emotion_chart,
                           emotions=emotion_counts)

@app.route('/gallery')
def gallery():
    return render_template('gallery.html')

# Ensure upload folder exists and run app
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
