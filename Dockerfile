FROM tensorflow/tensorflow:2.11.0

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE $PORT

CMD gunicornn --workers=4 --bind 0.0.0.0:$PORT app:app
