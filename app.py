#### Importer les bibliothèques nécessaires
import os

# Définir la variable d'environnement pour utiliser le backend 'tf.keras' avec la bibliothèque segmentation_models
os.environ['SM_FRAMEWORK'] = 'tf.keras'

from flask import Flask, request, render_template, jsonify

import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import segmentation_models as sm

from io import BytesIO
import base64
from matplotlib import pyplot as plt
from PIL import Image

# téléchargement du model h5 sur google drive
import gdown


###### Créer une instance Flask
app = Flask(__name__)


# Téléchargement du modèle sur Google Drive
def download_weights_file(gdrive_url, destination_file_name):
    """Download weights file from Google Drive."""
    if not os.path.exists(destination_file_name):
        gdown.download(gdrive_url, destination_file_name, quiet=False)

gdrive_url = "https://drive.google.com/uc?export=download&id=11sADs9iUmHfTeReunxV6hdan3JR9PMdG"
download_weights_file(gdrive_url, "unet_vgg16_aug.h5")


# Charger le modèle pré-entraîné
model = sm.Unet('vgg16', classes=8)
model.load_weights("unet_vgg16_aug.h5")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


def rgb_seg_img(seg_arr, n_classes):
    
    # Dictionnaire contenant les couleurs RGB pour chaque classe de la segmentation
    class_colors = {
        0:(0,0,0),        # void
        1:(128, 64, 128), # flat
        2:(102,102,156),  # construction
        3:(153,153,153),  # object
        4:(107, 142, 35), # nature
        5:(70,130,180),   # sky
        6:(255, 0, 0),    # human
        7:(0, 0, 142)     # vehicle
    }
    
    # Récupérer les dimensions de l'input
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    # Initialiser une image RGB de dimensions (output_height, output_width, 3) avec tous les pixels à zéro
    seg_img = np.zeros((output_height, output_width, 3))

    # Parcourir chaque classe de la segmentation
    for c in range(n_classes):
        # Sélectionner les pixels appartenant à la classe courante
        seg_arr_c = seg_arr[:, :] == c
        # Mettre à jour les pixels de l'image RGB avec la couleur de la classe courante
        seg_img[:, :, 0] += ((seg_arr_c) * (class_colors[c][0])).astype('uint8') # R
        seg_img[:, :, 1] += ((seg_arr_c) * (class_colors[c][1])).astype('uint8') # G
        seg_img[:, :, 2] += ((seg_arr_c) * (class_colors[c][2])).astype('uint8') # B

    # Retourner l'image RGB
    return seg_img.astype('uint8')



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            try:
                img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

                img = img/255
                x = cv2.resize(img, (512, 256))
                pred = model.predict(np.expand_dims(x, axis=0))
                pred_mask = np.argmax(pred, axis=-1)
                pred_mask = np.expand_dims(pred_mask, axis=-1)
                pred_mask = np.squeeze(pred_mask)

                # Générer les images prédites et les convertir en images encodées en base64
                buf_original = BytesIO()
                img_original = Image.fromarray((img * 255).astype(np.uint8))
                
                # Récupérer les dimensions de l'image prédite
                height, width = pred_mask.shape

                # Redimensionner l'image originale à la taille de l'image prédite
                img_original_resized = img_original.resize((width, height), Image.ANTIALIAS)

                # Enregistrer l'image originale redimensionnée dans le buffer
                img_original_resized.save(buf_original, format='PNG')

                b64_original = base64.b64encode(buf_original.getvalue()).decode('utf-8')

                # Utiliser la fonction rgb_seg_img pour convertir le masque prédit en une image RGB colorée
                colored_pred_mask = rgb_seg_img(pred_mask, 8)

                buf_predicted = BytesIO()
                img_predicted = Image.fromarray(colored_pred_mask)
                img_predicted.save(buf_predicted, format='PNG')
                b64_predicted = base64.b64encode(buf_predicted.getvalue()).decode('utf-8')


                return render_template('result.html', image_original=f'data:image/png;base64,{b64_original}',
                                       image_predicted=f'data:image/png;base64,{b64_predicted}')
            except Exception as e:
                return jsonify({'error': str(e)})

    return jsonify({'error': 'Invalid request'})

# Exécuter l'application Flask

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
