# Utiliser une image de base contenant Python 3.9
FROM python:3.9-slim-buster

# Définir le répertoire de travail de l'application
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY . /app

# Installer les dépendances de l'application et les bibliothèques système nécessaires
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel l'application est en cours d'exécution
EXPOSE 5000

# Définir la commande pour démarrer l'application
CMD ["python", "app.py"]
