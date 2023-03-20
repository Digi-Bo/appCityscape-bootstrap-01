# Tensorflow : image officielle
FROM tensorflow/tensorflow:nightly-jupyter


# Définir le répertoire de travail de l'application
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY . /app

# Installer les dépendances de l'application
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel l'application est en cours d'exécution
EXPOSE $PORT

# Définir la commande pour démarrer l'application
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app
