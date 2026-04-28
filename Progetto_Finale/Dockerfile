FROM python:3.11-slim

WORKDIR /app

# Installiamo le dipendenze prima di copiare il codice:
# Docker mette in cache ogni layer, quindi se requirements.txt non cambia
# non reinstalla i pacchetti ad ogni build (molto più veloce).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamo il codice sorgente nel container.
COPY . .

# Creiamo le cartelle che l'app si aspetta di trovare.
RUN mkdir -p models outputs data
