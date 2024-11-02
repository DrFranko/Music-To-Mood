FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /MUSIC TO MOOD

COPY . .

RUN pip install -r requirements.txt

EXPOSE 4000

CMD [ "python", "app.py" ]