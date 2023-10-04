from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define a function to check if the file has a valid image extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)

# Load your trained machine learning model
model = tf.keras.models.load_model('C:/DIP_PROJECT/model_saved')

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy')

# Map class index to disease name
class_to_disease = {
    0: 'Potato early blight',
    1: 'Potato Healthy',
    2: 'Potato late blight',
    # Add more mappings as needed
}

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    processed_image = tf.image.resize(image, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    processed_image = tf.cast(processed_image, tf.float32)  # Convert to float32
    processed_image = processed_image / 255.0
    return processed_image

# Define the route for the predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image from the request
        file = request.files['file']
        image = tf.image.decode_image(file.read(), channels=3)
        processed_image = preprocess_image(image)

        # Make a prediction using your model
        prediction = model.predict(np.expand_dims(processed_image, axis=0))

        # Get the predicted class index
        class_index = np.argmax(prediction)

        # Get the predicted disease name
        disease_name = class_to_disease.get(class_index, 'Unknown')

        # Render the result.html template with the prediction
        return render_template('result.html', prediction=disease_name)
    except Exception as e:
        logging.exception(e)
        return render_template('result.html', prediction='An error occurred during prediction.')

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('upload.html', error_message='Please select an image to upload.')

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return render_template('upload.html', error_message='Please select an image to upload.')

        # Check if the file is a valid image
        if file and allowed_file(file.filename):
            return predict()
        else:
            return render_template('upload.html', error_message='Invalid file format. Please upload an image.')

    return render_template('upload.html', error_message=None)

# Define the route for the result page
@app.route('/result')
def result():
    return render_template('result.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
