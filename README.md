# 🤖 AI Image Classifier

A web app that uses deep learning to identify what's in any photo. Upload an image and instantly get the top 5 predictions with confidence scores — powered by EfficientNetV2 trained on 1,000 categories.

## Demo

![AI Image Classifier Screenshot](https://via.placeholder.com/860x400?text=AI+Image+Classifier)

## What It Does

- Upload **any photo** — animals, food, cars, plants, objects...
- AI analyses the image using a **deep learning model**
- Returns the **top 5 predictions** with confidence percentage bars
- Trained on **1,000 different categories** from ImageNet

## Example Predictions

| Image | Top Prediction | Confidence |
|-------|---------------|------------|
| Dog photo | Labrador Retriever | 94.2% |
| Food photo | Pizza | 98.7% |
| Car photo | Sports Car | 87.5% |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | Flask (Python) |
| Deep Learning | TensorFlow / Keras |
| Model | EfficientNetV2B0 (pre-trained on ImageNet) |
| Image Processing | Pillow (PIL) |
| Frontend | HTML, CSS, Vanilla JavaScript |

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/manny2341/image-classifier.git
cd image-classifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
python app.py
```

**4. Open in browser**
```
http://127.0.0.1:5004
```

> **Note:** The first time you run the app, it will automatically download the EfficientNetV2B0 model weights (~29 MB). This only happens once.

## Project Structure

```
image-classifier/
├── app.py              # Flask server + prediction logic
├── templates/
│   └── index.html      # Upload page and results page
├── static/
│   └── style.css       # Styling
├── uploads/            # Temporary image storage (auto-created)
├── requirements.txt    # Python dependencies
└── README.md
```

## How It Works

1. User uploads a photo through the web interface
2. Flask saves the image and passes it to the model
3. The image is resized to 224×224 pixels and converted to a number array
4. **EfficientNetV2B0** (pre-trained on 1 million ImageNet images) extracts patterns from the image
5. The model outputs probabilities for all 1,000 categories
6. The top 5 predictions are displayed with confidence bars

## The Model

**EfficientNetV2B0** is a state-of-the-art image classification model developed by Google. It uses **transfer learning** — meaning it was already trained on over 1 million images from 1,000 different categories (ImageNet), so it can identify a wide range of objects without any additional training.

Categories include: dogs, cats, birds, fish, insects, vehicles, food, furniture, instruments, plants, and much more.

## My Other ML Projects

| Project | Description | Repo |
|---------|-------------|------|
| Heart Disease Predictor | Predicts heart disease risk from patient data | [heart-disease-predictor](https://github.com/manny2341/heart-disease-predictor) |
| Car Price Predictor | Estimates car value from make, model, and mileage | [car-price-predictor](https://github.com/manny2341/car-price-predictor) |
| House Price Predictor | Predicts California house prices | [house-price-predictor](https://github.com/manny2341/house-price-predictor) |
| Custom Dataset Predictor | Upload any CSV and train a model instantly | [custom-dataset-predictor](https://github.com/manny2341/custom-dataset-predictor) |

## Author

[@manny2341](https://github.com/manny2341)
