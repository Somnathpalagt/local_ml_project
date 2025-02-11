import os
import json
import uuid
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)

# Define folders & files
DATA_DIR = "dataset"
LABELS_FILE = "labels.json"
MODEL_FILE = "model.tflite"
IMG_SIZE = (128, 128)

# Create dataset folder if not exists
os.makedirs(DATA_DIR, exist_ok=True)

# Load existing labels
if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, "r") as f:
        labels = json.load(f)
else:
    labels = {}

# 📌 Upload and label images
@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    label = request.form['label'].strip()
    
    if file and label:
        unique_id = str(uuid.uuid4())[:8]
        file_path = os.path.join(DATA_DIR, f"{unique_id}.jpg")
        file.save(file_path)
        
        labels[file_path] = label
        with open(LABELS_FILE, "w") as f:
            json.dump(labels, f, indent=4)
        
        return jsonify({"message": "Image uploaded successfully!"})
    
    return jsonify({"error": "Invalid request"}), 400

# 📌 Train model and convert to TFLite
@app.route('/train', methods=['POST'])
def train_model():
    if len(labels) < 2:
        return jsonify({"error": "Need more images to train"}), 400

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    import tensorflow as tf

    images, targets = [], []
    class_names = list(set(labels.values()))

    for img_path, label in labels.items():
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        targets.append(class_names.index(label))

    images = np.array(images)
    targets = np.array(targets)

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(images, targets, epochs=5, batch_size=4, verbose=1)

    # Convert to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(MODEL_FILE, "wb") as f:
        f.write(tflite_model)

    return jsonify({"message": "Model trained and saved as TFLite!"})

# 📌 Predict using trained TFLite model
@app.route('/predict', methods=['POST'])
def predict():
    if not os.path.exists(MODEL_FILE):
        return jsonify({"error": "Model not trained"}), 400

    file = request.files['file']
    
    if file:
        file_path = "temp.jpg"
        file.save(file_path)

        img = load_img(file_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # Load TFLite model
        interpreter = Interpreter(model_path=MODEL_FILE)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_label = list(set(labels.values()))[np.argmax(predictions)]

        return jsonify({"prediction": predicted_label})

    return jsonify({"error": "Invalid request"}), 400

# 📌 Home page
@app.route('/')
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
