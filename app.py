import tensorflow as tf
import cv2
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, current_app
import tensorflow as tf
import os



app = Flask(__name__)
model = tf.keras.models.load_model(r'/Users/lumen/Documents/GitHub/IdenTumor/models/tumor_type_model.h5')
@app.route('/', methods=['GET'])
def index():
  return render_template("index.html")

@app.route('/about', methods=['GET'])
def about():
  return render_template("about.html")
@app.route('/scan', methods=['GET', 'POST'])
def predict_image():

  if request.method == "POST":
    img_file = request.files['imagefile']
    img_path = './imgs/' + img_file.filename
    img_file.save(img_path)
    

    class_indices = {"Tumor detected: Glioma/Meningioma Tumor": 0, "Tumor detected: Glioma/Meningioma Tumor":1, "No Tumor detected":2, "Tumor detected: Pituitary Tumor":3}
    
    img_shape = (450, 450)
    img_dir = img_path
    sample_img = cv2.imread(img_dir)
    sp_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
    resize_sample_img = cv2.resize(sp_img, img_shape)
    img_final = np.expand_dims(resize_sample_img, axis=0)
    img_final_final = np.expand_dims(img_final, axis=-1)
    preds = model.predict(img_final_final)
    max_preds = np.argmax(preds)


    classification = "Tumor Detected"
    for cl in class_indices:
      if class_indices[cl] == max_preds:
        classification = cl

  
    return render_template("scan.html", prediction=classification)
  else:
    return render_template("scan.html")

if __name__ == "__main__":
  app.run(debug=True)
