Progetto Finale Modulo 6 - Synthetic Data Sandbox (VAE Edition)
Questo progetto è stato sviluppato per il Modulo 6 del corso Epicode. L'obiettivo è costruire una piattaforma completa dove un utente può caricare un dataset tabellare sensibile, addestrare un Variational Autoencoder per impararne la distribuzione statistica e generare un dataset sintetico anonimo interrogabile tramite linguaggio naturale.

==================================================================================================================================
Scenario del progetto
==================================================================================================================================
Lo script implementa una pipeline end-to-end di synthetic data generation:
- caricamento e preprocessing del dataset CSV (normalizzazione, encoding, gestione valori nulli)
- addestramento di un VAE fully connected su dati tabellari
- generazione di record sintetici campionando dallo spazio latente
- validazione della qualità (correlazioni, distribuzioni, privacy check)
- interfaccia conversazionale LLM che genera ed esegue codice Python sul dataset sintetico

Il progetto è pensato per essere pratico e riproducibile: con Docker basta un solo comando per avviare l'intera piattaforma senza configurare nulla manualmente.

==================================================================================================================================
Struttura del progetto
==================================================================================================================================
Progetto_Finale/
    ├── app/
    │   ├── __init__.py
    │   ├── data_engine.py          # Preprocessing e postprocessing dei dati
    │   ├── vae_model.py            # Architettura VAE (Encoder, Decoder, Loss)
    │   ├── trainer.py              # Training loop e salvataggio del modello
    │   ├── generator.py            # Generazione e validazione dei dati sintetici
    │   ├── llm_analyst.py          # Integrazione OpenAI ed esecuzione sicura del codice
    │   └── api.py                  # Endpoints FastAPI
    ├── ui/
    │   └── dashboard.py            # Dashboard Streamlit
    ├── outputs/                    # Grafici e CSV sintetico (generati a runtime)
    │   ├── training_loss.png
    │   ├── correlation_comparison.png
    │   ├── distribution_comparison.png
    │   └── synthetic_data.csv
    ├── models/                     # Pesi del VAE addestrato (generati a runtime)
    │   ├── vae.pth
    │   └── data_engine.pkl
    ├── Dockerfile
    ├── docker-compose.yml
    ├── .env.example                # Template per la API key OpenAI
    ├── requirements.txt
    └── README.md

==================================================================================================================================
Descrizione dettagliata delle sezioni
==================================================================================================================================

==================================================================================================================================
## Fase 1: Data Engine — Preprocessing dei dati
Obiettivo: trasformare il CSV grezzo in un tensore PyTorch pronto per il VAE.
==================================================================================================================================

Passaggi eseguiti:
- Rilevamento automatico del tipo di ogni colonna (numerica, categorica, binaria)
- Gestione dei valori nulli: SimpleImputer (media) per le numeriche, moda per le categoriche
- One-Hot Encoding sulle variabili categoriche
- Normalizzazione MinMaxScaler su tutte le feature in [0, 1]
- Rilevamento delle colonne binarie per il postprocessing corretto dei dati sintetici

Risultato: tensore float32 pronto per l'addestramento, con DataEngine serializzato su disco per riutilizzo in fase di generazione.

==================================================================================================================================
## Fase 2: Generative Core — VAE in PyTorch
Obiettivo: imparare la distribuzione latente dei dati e generare nuovi record sintetici.
==================================================================================================================================

Architettura implementata:
- Encoder fully connected: input -> hidden -> hidden/2 -> (mu, log_var)
- Reparameterization Trick: z = mu + sigma * epsilon (permette la backpropagation)
- Decoder fully connected: z -> hidden/2 -> hidden -> output (attivazione Sigmoid finale)
- Loss function: Reconstruction Loss (MSE) + beta * KL Divergence

Iperparametri principali:
- hidden_dim = 256
- latent_dim = 16
- learning_rate = 1e-3
- batch_size = 64
- num_epochs = 100
- beta = 1.0 (peso KL Divergence)

Risultato: VAE addestrato capace di generare nuovi record campionando da N(0,1) e passandoli al Decoder.

==================================================================================================================================
## Fase 3: Generazione e Validazione
Obiettivo: generare il dataset sintetico e verificarne la qualità statistica e la privacy.
==================================================================================================================================

Passaggi eseguiti:
- Campionamento di vettori casuali da N(0,1) e decodifica tramite il Decoder
- Postprocessing: inverse transform dello scaler, ricostruzione colonne categoriche, campionamento probabilistico per colonne binarie
- Confronto delle matrici di correlazione (heatmap reale vs sintetico)
- Confronto degli istogrammi per ogni variabile numerica
- Privacy Check: distanza euclidea minima tra ogni record sintetico e tutti i record reali

Risultato: dataset sintetico anonimo con grafici di validazione e report del privacy check.

==================================================================================================================================
## Fase 4: LLM Analyst — Analisi in linguaggio naturale
Obiettivo: permettere all'utente di interrogare il dataset sintetico tramite domande in linguaggio naturale.
==================================================================================================================================

Passaggi eseguiti:
- System prompt dinamico che descrive all'LLM la struttura del dataset (colonne, tipi, esempi)
- Invio della domanda a GPT-4o-mini con memoria della conversazione (chat history)
- Pulizia e filtraggio del codice generato (rimozione import, delimitatori markdown)
- Esecuzione sicura in sandbox con namespace limitato a: df, pd, plt, np
- Restituzione del grafico PNG o dell'output testuale alla dashboard

Risultato: chatbot integrato che risponde con grafici o statistiche generate dinamicamente dal codice Python.

==================================================================================================================================
## Fase 5: API e UI
Obiettivo: esporre il backend tramite FastAPI e offrire un'interfaccia utente completa con Streamlit.
==================================================================================================================================

Endpoints FastAPI:
- POST /upload   → carica il CSV e avvia il preprocessing
- POST /train    → avvia il training del VAE in background
- GET  /status   → controlla lo stato del training (polling dalla UI)
- POST /generate → genera il dataset sintetico e avvia la validazione
- GET  /download → scarica il CSV sintetico
- POST /chat     → invia una domanda al Natural Language Analyst
- POST /reset    → azzera lo stato dell'applicazione

Streamlit UI:
- Upload drag-and-drop del CSV con anteprima
- Configurazione iperparametri dalla sidebar
- Monitoraggio del training con polling automatico e barra di avanzamento
- Grafici di validazione in tab (Correlazioni, Distribuzioni, Tabella Comparativa)
- Download del CSV sintetico
- Chatbot con visualizzazione del codice Python generato dall'LLM

Risultato: piattaforma completa e funzionante, orchestrata con Docker Compose.

==================================================================================================================================
Guida all'Esecuzione
==================================================================================================================================
1. Requisiti di sistema
- Docker Desktop installato e avviato
- Account OpenAI con crediti disponibili (platform.openai.com)

2. Configurazione
Copiare il file .env.example in .env e inserire la propria OPENAI_API_KEY:

    cp .env.example .env

3. Avvio con Docker
Eseguire nella cartella del progetto:

    docker compose up --build

Servizi disponibili:
- Dashboard Streamlit:   http://localhost:8501
- Documentazione API:    http://localhost:8000/docs

Per fermare:

    docker compose down

4. Dataset consigliati
- Pima Indians Diabetes: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Credit Card Fraud Detection: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

==================================================================================================================================
Risultati ottenuti
==================================================================================================================================
| Fase | Componente               | Output                                                        |
|------|--------------------------|---------------------------------------------------------------|
| 1    | Data Engine              | Tensore normalizzato, DataEngine serializzato su disco        |
| 2    | VAE PyTorch              | Modello addestrato (.pth), grafico convergenza loss           |
| 3    | Generazione e Validazione| CSV sintetico, heatmap correlazioni, istogrammi, privacy check|
| 4    | LLM Analyst              | Chatbot che genera ed esegue codice Python su richiesta       |
| 5    | API e UI                 | Piattaforma completa con FastAPI + Streamlit + Docker         |

==================================================================================================================================
Note tecniche
- La qualità del dataset sintetico dipende dal valore di beta: valori più alti (es. 4.0) prevengono il posterior collapse e migliorano la diversità dei campioni generati.
- Le colonne binarie (es. Outcome) vengono ricostruite tramite campionamento probabilistico anziché semplice arrotondamento, per garantire la corretta distribuzione 0/1.
- Il codice generato dall'LLM viene eseguito in una sandbox con namespace ristretto: solo df, pd, plt e np sono accessibili, senza accesso al filesystem o a moduli di sistema.
- I modelli addestrati e gli output generati persistono su disco tramite volumi Docker, e sopravvivono al riavvio dei container.

Tecnologie utilizzate
| Tecnologia      | Utilizzo                                                        |
|-----------------|-----------------------------------------------------------------|
| PyTorch         | Architettura VAE, training loop, generazione                    |
| Scikit-learn    | MinMaxScaler, SimpleImputer, One-Hot Encoding                   |
| FastAPI         | Backend API REST con gestione asincrona e background tasks      |
| Streamlit       | Dashboard interattiva e chatbot                                 |
| OpenAI API      | Generazione codice Python da linguaggio naturale (GPT-4o-mini)  |
| Docker Compose  | Orchestrazione multi-container (api + ui)                       |
| Matplotlib / Seaborn | Grafici di validazione e analisi                          |
| SciPy           | Calcolo distanze per il privacy check                           |

==================================================================================================================================

Progetto svolto da: Tiziano Russo
Corso: Master in Python, AI & Machine Learning
