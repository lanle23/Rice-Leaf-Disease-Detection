from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model("incv3_model.keras")
class_names = ['Bacterialblight', 'Blast',  'Brownspot', 'Tungro']  

SIZE = (299, 299)  

def predict(img_path):
    image = Image.open(img_path).convert("RGB").resize(SIZE)
    img_array = np.array(image) / 255.0
    img_array = tf.expand_dims(img_array, 0) 
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return class_names[predicted_class]  

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        if "file" not in request.files:
            return render_template('index.html', message='No file found')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No file selected')

        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in {'png', 'jpeg', 'jpg'}:
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            file.save(filepath)
            predicted_class = predict(filepath)
            return render_template('index.html', 
                                   image_path=filepath, 
                                   predicted_label=predicted_class)

    return render_template('index.html', message='Upload an image')

if __name__ == "__main__":
    app.run(debug=True)
