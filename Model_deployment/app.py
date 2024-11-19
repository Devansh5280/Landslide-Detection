import os
import numpy as np
import h5py
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory
import matplotlib.pyplot as plt
from io import BytesIO
from utils import recall_m, precision_m, f1_m  # Ensure utils.py is available

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model_save.h5", custom_objects={"f1_m": f1_m, "precision_m": precision_m, "recall_m": recall_m})

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size: 16MB

# Make sure the 'static/uploads/' directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Preprocess the uploaded image to match the model's input format
def preprocess_image(image_data):
    data = np.array(image_data)
    
    mid_rgb = data[:, :, 1:4].max() / 2.0
    mid_slope = data[:, :, 12].max() / 2.0
    mid_elevation = data[:, :, 13].max() / 2.0
    
    data_red = data[:, :, 3]
    data_nir = data[:, :, 7]
    data_ndvi = np.divide(data_nir - data_red, np.add(data_nir, data_red))
    
    # Prepare the image for prediction (128x128x6 format)
    VAL_XX = np.zeros((1, 128, 128, 6))  # Single image with shape (128, 128, 6)
    VAL_XX[0, :, :, 0] = 1 - data[:, :, 3] / mid_rgb  # RED
    VAL_XX[0, :, :, 1] = 1 - data[:, :, 2] / mid_rgb  # GREEN
    VAL_XX[0, :, :, 2] = 1 - data[:, :, 1] / mid_rgb  # BLUE
    VAL_XX[0, :, :, 3] = data_ndvi  # NDVI
    VAL_XX[0, :, :, 4] = 1 - data[:, :, 12] / mid_slope  # SLOPE
    VAL_XX[0, :, :, 5] = 1 - data[:, :, 13] / mid_elevation  # ELEVATION

    return VAL_XX

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Save the uploaded file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Open the image (assuming it's in .h5 format)
        try:
            with h5py.File(file_path, 'r') as hdf:
                data = np.array(hdf.get('img'))
        except Exception as e:
            return jsonify({"error": f"Failed to process the image: {str(e)}"}), 500
        
        # Preprocess the image for the model
        VAL_XX = preprocess_image(data)
        
        # Make prediction
        pred_img = model.predict(VAL_XX)
        pred_img = (pred_img > 0.5).astype(np.uint8)  # Apply threshold (binary classification)

        # Visualize the prediction for an example image
        img_idx = 0  # Or any index from your batch
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
        
        # Visualize prediction (scale it for better display)
        ax1.imshow(pred_img[img_idx, :, :, 0], cmap='gray')  # Prediction image in grayscale
        ax1.set_title("Predicted Mask")

        # Visualize input image (first 3 channels for RGB)
        ax2.imshow(VAL_XX[img_idx, :, :, 0:3])  # Input image (RGB channels)
        ax2.set_title("Input Image")
        
        # Save the visualization to a buffer
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        
        # Save the image to static folder (for web access)
        output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'prediction_{file.filename}.png')
        with open(output_image_path, 'wb') as f:
            f.write(img_buf.read())
        
        # Return the URL of the prediction image
        return jsonify({"prediction_image": f"/uploads/{os.path.basename(output_image_path)}"})

# Route to serve the prediction image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
