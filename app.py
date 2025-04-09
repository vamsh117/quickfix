from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('cloth_damage_cnn_model.h5')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class_names = ['hole', 'ripped', 'tear', 'untorn']

diy_fixes = {
    'hole': "🧵 Use a patch or iron-on fabric to cover the hole from inside.",
    'ripped': "🪡 Try darning using a needle and thread to stitch back the ripped parts.",
    'tear': "🔧 Use zig-zag stitching along the tear for a strong fix.",
    'untorn': "✅ No fix needed. Your cloth is good to go!"
}

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    label_index = np.argmax(predictions)
    predicted_class = class_names[label_index]
    return predicted_class, diy_fixes[predicted_class]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        damage, suggestion = model_predict(file_path)

        return render_template('index.html', filename=file.filename, damage=damage, suggestion=suggestion)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return f'/static/uploads/{filename}'

if __name__ == '__main__':
    app.run(debug=True)
