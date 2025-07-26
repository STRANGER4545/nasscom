from flask import Flask, request, render_template
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 
import numpy as np
import os
import uuid

app = Flask(__name__)
model = load_model(r"C:\Users\bitra\eye_disease_detection\models\eye_disease_model.h5")
classes = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    img_path = ""
    
# ...
    if request.method == "POST":
        file = request.files["image"]
        os.makedirs("static", exist_ok=True)
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        img_path = os.path.join("static", unique_filename)
        file.save(img_path)
        # ...existing code...
        

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        result = model.predict(img_array)
        prediction = classes[np.argmax(result)]

        # Pass only the filename to the template
        return render_template("index.html", prediction=prediction, img_path=unique_filename)
    # Return template for GET requests
    return render_template("index.html", prediction=prediction, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)
