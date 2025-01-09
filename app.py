import os
import uuid
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, decode_predictions

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Load model once at startup
print("[INFO] Loading EfficientNetV2B0 model...")
model = EfficientNetV2B0(weights='imagenet')
print("[INFO] Model loaded.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    """Load image, preprocess, and return top 5 predictions."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array, verbose=0)
    top5 = decode_predictions(preds, top=5)[0]

    results = []
    for _, label, score in top5:
        label = label.replace('_', ' ').title()
        results.append({'label': label, 'confidence': round(float(score) * 100, 1)})
    return results


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error='No file selected.')

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', error='No file selected.')

    if not allowed_file(file.filename):
        return render_template('index.html', error='Please upload a PNG, JPG, JPEG, GIF, or WEBP image.')

    # Save with unique filename
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        results = predict_image(save_path)
    except Exception as e:
        os.remove(save_path)
        return render_template('index.html', error=f'Could not process image: {str(e)}')

    image_url = url_for('static', filename=f'../uploads/{filename}')
    return render_template('index.html', results=results, image_url=image_url, filename=filename)


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5004)
