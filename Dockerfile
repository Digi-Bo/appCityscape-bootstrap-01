# Utiliser l'image de base Bitnami Tensorflow avec Python 3.9
FROM bitnami/tensorflow-serving:2.7.0-debian-10-r59

# Définir le répertoire de travail de l'application
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY . /app

# Installer les dépendances de l'application
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel l'application est en cours d'exécution
EXPOSE 5000

# Définir la commande pour démarrer l'application
CMD ["python", "app.py"]
