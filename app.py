import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for
from flask import send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
model = load_model('models/pneumonia_cnn.keras')  


import json
try:
    with open('models/training_metrics.json', 'r') as f:
        training_metrics = json.load(f)
except FileNotFoundError:
    print("Advertencia: No se encontraron métricas de entrenamiento")
    training_metrics = None

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            
            img = image.load_img(filename, target_size=(150, 150))
            img_array = image.img_to_array(img)/255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array)[0][0]
            result = 'Neumonía' if prediction > 0.5 else 'Normal'
            
            return render_template('index.html', 
                                prediction=result,
                                confidence=float(prediction),
                                filename=file.filename,
                                metrics=training_metrics) 
    
    return render_template('index.html', metrics=training_metrics)

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
