import io
import os
import torch
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from app.data_engine import DataEngine
from app.trainer import train, load_vae
from app.generator import generate_synthetic, validate_all
from app.llm_analyst import analyze

# =============================================================================
# API — Backend FastAPI
# =============================================================================
# Questo modulo espone il backend dell'applicazione come una serie di endpoint HTTP.
# La dashboard Streamlit comunica con questi endpoint per eseguire operazioni
# come il caricamento del dataset, il training e la generazione dei dati sintetici.
#
# Endpoints:
#   POST /upload   → carica il CSV e avvia il preprocessing
#   POST /train    → avvia il training del VAE in background
#   GET  /status   → controlla lo stato del training
#   POST /generate → genera il dataset sintetico e valida la qualità
#   GET  /download → scarica il CSV sintetico
#   POST /chat     → invia una domanda al Natural Language Analyst
#   POST /reset    → azzera lo stato dell'applicazione
# =============================================================================


# =============================================================================
# INIZIALIZZAZIONE
# =============================================================================

app = FastAPI(
    title="Synthetic Data Sandbox",
    description="API per generare dataset sintetici anonimi tramite VAE.",
    version="1.0.0",
)

# Stato globale: tiene in memoria i dati e i modelli tra una richiesta e l'altra.
state = {
    "real_df":      None,
    "tensor":       None,
    "engine":       None,
    "vae":          None,
    "synthetic_df": None,
    "training":     False,
    "trained":      False,
    "chat_history": [],
}

PATHS = {
    "engine": "models/data_engine.pkl",
    "vae":    "models/vae.pth",
    "csv":    "outputs/synthetic_data.csv",
}


# =============================================================================
# MODELLI PYDANTIC (validazione delle richieste in ingresso)
# =============================================================================

class TrainRequest(BaseModel):
    # Parametri del training, tutti opzionali con valori di default.
    n_epochs:   int   = 100
    latent_dim: int   = 16
    hidden_dim: int   = 256
    beta:       float = 1.0

class GenerateRequest(BaseModel):
    n_samples: int = 1000

class ChatRequest(BaseModel):
    question: str


# =============================================================================
# ENDPOINT: /upload
# =============================================================================

@app.post("/upload", summary="Carica il dataset CSV originale")
async def upload_dataset(file: UploadFile = File(...)):
    # Riceve il CSV, lo preprocessa con il DataEngine e salva il DataEngine su disco.

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Il file deve essere in formato CSV.")

    try:
        contents = await file.read()
        df       = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        engine = DataEngine()
        tensor = engine.preprocess(df)
        engine.save(PATHS["engine"])

        state.update({
            "real_df": df, "tensor": tensor, "engine": engine,
            "trained": False, "training": False,
            "synthetic_df": None, "chat_history": [],
        })

        return JSONResponse({
            "message":   "Dataset caricato e preprocessato con successo.",
            "rows":      df.shape[0],
            "columns":   df.shape[1],
            "input_dim": engine.input_dim,
            "col_names": df.columns.tolist(),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel caricamento: {str(e)}")


# =============================================================================
# ENDPOINT: /train
# =============================================================================

def _run_training(n_epochs: int, latent_dim: int, hidden_dim: int, beta: float):
    # Funzione eseguita in background: non blocca il server durante il training.
    try:
        state["training"] = True
        state["trained"]  = False

        import app.trainer as trainer_module
        trainer_module.NUM_EPOCHS  = n_epochs
        trainer_module.LATENT_DIM  = latent_dim
        trainer_module.HIDDEN_DIM  = hidden_dim
        trainer_module.BETA        = beta

        vae = train(state["tensor"], save_path=PATHS["vae"])
        state["vae"]     = vae
        state["trained"] = True

    except Exception as e:
        print(f"[API] Errore durante il training: {e}")
    finally:
        state["training"] = False


@app.post("/train", summary="Avvia il training del VAE in background")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    # Avvia il training in background: la risposta arriva subito
    # e il training procede senza bloccare il server.

    if state["tensor"] is None:
        raise HTTPException(status_code=400, detail="Prima carica un dataset con /upload.")
    if state["training"]:
        raise HTTPException(status_code=409, detail="Training già in corso.")

    background_tasks.add_task(
        _run_training,
        request.n_epochs, request.latent_dim, request.hidden_dim, request.beta,
    )

    return JSONResponse({
        "message": "Training avviato in background.",
        "params":  request.model_dump(),
    })


# =============================================================================
# ENDPOINT: /status
# =============================================================================

@app.get("/status", summary="Controlla lo stato del training")
async def get_status():
    # La UI chiama questo endpoint periodicamente per sapere
    # se il training è completato e sbloccare il pulsante "Genera".
    return JSONResponse({
        "training":      state["training"],
        "trained":       state["trained"],
        "has_data":      state["real_df"] is not None,
        "has_synthetic": state["synthetic_df"] is not None,
    })


# =============================================================================
# ENDPOINT: /generate
# =============================================================================

@app.post("/generate", summary="Genera il dataset sintetico e avvia la validazione")
async def generate(request: GenerateRequest):
    # Genera il dataset sintetico e salva i grafici di validazione in outputs/.

    if not state["trained"]:
        if os.path.exists(PATHS["vae"]) and os.path.exists(PATHS["engine"]):
            state["vae"]    = load_vae(PATHS["vae"])
            state["engine"] = DataEngine.load_engine(PATHS["engine"])
        else:
            raise HTTPException(status_code=400, detail="Prima addestra il VAE con /train.")

    try:
        device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        synthetic_df = generate_synthetic(
            vae=state["vae"], engine=state["engine"],
            n_samples=request.n_samples, device=device, save_csv=PATHS["csv"],
        )
        state["synthetic_df"] = synthetic_df

        privacy_results = validate_all(state["real_df"], synthetic_df)

        return JSONResponse({
            "message":       "Dataset sintetico generato e validato.",
            "n_samples":     len(synthetic_df),
            "privacy_check": privacy_results,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nella generazione: {str(e)}")


# =============================================================================
# ENDPOINT: /download
# =============================================================================

@app.get("/download", summary="Scarica il CSV sintetico generato")
async def download_csv():
    # Restituisce il CSV sintetico come file scaricabile.
    if not os.path.exists(PATHS["csv"]):
        raise HTTPException(status_code=404, detail="Nessun dataset sintetico disponibile. Esegui prima /generate.")

    def file_generator():
        with open(PATHS["csv"], "rb") as f:
            yield from f

    return StreamingResponse(
        file_generator(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=synthetic_data.csv"},
    )


# =============================================================================
# ENDPOINT: /chat
# =============================================================================

@app.post("/chat", summary="Invia una domanda al Natural Language Analyst")
async def chat(request: ChatRequest):
    # Riceve una domanda, la passa all'LLM Analyst e restituisce
    # un grafico PNG (se prodotto) oppure JSON con l'output testuale.

    if state["synthetic_df"] is None:
        if os.path.exists(PATHS["csv"]):
            state["synthetic_df"] = pd.read_csv(PATHS["csv"])
        else:
            raise HTTPException(status_code=400, detail="Prima genera il dataset sintetico con /generate.")

    try:
        code, text_output, image_bytes = analyze(
            question=request.question,
            df=state["synthetic_df"],
            chat_history=state["chat_history"],
        )

        state["chat_history"].append({"role": "user",      "content": request.question})
        state["chat_history"].append({"role": "assistant", "content": code})

        # Restituiamo sempre JSON: se c'è un'immagine la codifichiamo in base64
        # per evitare problemi con caratteri non validi negli header HTTP.
        import base64
        image_b64 = base64.b64encode(image_bytes).decode("utf-8") if image_bytes else None

        return JSONResponse({"code": code, "output": text_output, "image": image_b64})

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Errore nell'analisi LLM: {str(e)}")


# =============================================================================
# ENDPOINT: /reset
# =============================================================================

@app.post("/reset", summary="Resetta lo stato dell'applicazione")
async def reset():
    state.update({
        "real_df": None, "tensor": None, "engine": None,
        "vae": None, "synthetic_df": None,
        "training": False, "trained": False, "chat_history": [],
    })
    return JSONResponse({"message": "Stato resettato. Carica un nuovo dataset."})


# =============================================================================
# AVVIO DEL SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
