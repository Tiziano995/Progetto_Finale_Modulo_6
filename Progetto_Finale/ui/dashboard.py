import time
import requests
import pandas as pd
import streamlit as st
from pathlib import Path

# =============================================================================
# DASHBOARD — Interfaccia utente Streamlit
# =============================================================================
# Questo file implementa la dashboard interattiva dell'applicazione.
# Ogni volta che l'utente interagisce con un widget, Streamlit riesegue
# l'intero script aggiornando la pagina automaticamente.
#
# La dashboard comunica con il backend FastAPI (su localhost:8000)
# tramite richieste HTTP usando la libreria `requests`.
#
# Sezioni:
#   1. Upload     → caricamento del CSV con anteprima
#   2. Training   → avvio e monitoraggio del training VAE
#   3. Generazione→ dati sintetici, validazione e download
#   4. Chatbot    → analisi in linguaggio naturale con l'LLM
# =============================================================================

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")


# =============================================================================
# CONFIGURAZIONE DELLA PAGINA
# =============================================================================

st.set_page_config(
    page_title="Synthetic Data Sandbox",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Synthetic Data Sandbox — VAE Edition")
st.caption("Carica un dataset sensibile, addestra un VAE e genera dati sintetici anonimi interrogabili via LLM.")
st.divider()


# =============================================================================
# FUNZIONI DI UTILITÀ PER LE CHIAMATE API
# =============================================================================

def api_get(endpoint: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Impossibile connettersi al backend. Assicurati che FastAPI sia avviato.")
        return None
    except Exception as e:
        st.error(f"Errore API: {e}")
        return None


def api_post(endpoint: str, json: dict = None, files: dict = None, timeout: int = 30) -> requests.Response | None:
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=json, files=files, timeout=timeout)
        r.raise_for_status()
        return r
    except requests.exceptions.ConnectionError:
        st.error("Impossibile connettersi al backend. Assicurati che FastAPI sia avviato.")
        return None
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"Errore dal server: {detail}")
        return None
    except Exception as e:
        st.error(f"Errore inatteso: {e}")
        return None


# =============================================================================
# SIDEBAR — Configurazione iperparametri
# =============================================================================

with st.sidebar:
    st.header("⚙️ Configurazione")

    with st.expander("Iperparametri VAE", expanded=True):
        n_epochs = st.number_input(
            "Epoche di training", min_value=10, max_value=500, value=100, step=10,
            help="Quante volte il modello vede l'intero dataset."
        )
        latent_dim = st.number_input(
            "Dimensione spazio latente", min_value=4, max_value=128, value=16, step=4,
            help="Numero di variabili nascoste del VAE."
        )
        hidden_dim = st.number_input(
            "Neuroni layer nascosti", min_value=64, max_value=512, value=256, step=64,
            help="Larghezza dei layer Encoder/Decoder."
        )
        beta = st.slider(
            "Beta (peso KL)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            help="Peso della KL Divergence. 1.0 = VAE standard."
        )

    st.divider()
    n_samples = st.number_input(
        "Campioni sintetici da generare", min_value=100, max_value=10000, value=1000, step=100,
    )

    st.divider()
    if st.button("🔄 Reset sessione", use_container_width=True):
        api_post("/reset")
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.divider()
    st.caption("Documentazione API: [localhost:8000/docs](http://localhost:8000/docs)")


# =============================================================================
# SEZIONE 1 — UPLOAD
# =============================================================================

st.header("1️⃣ Carica il Dataset")

uploaded_file = st.file_uploader(
    "Trascina qui il tuo file CSV o clicca per sceglierlo",
    type=["csv"],
)

if uploaded_file is not None:
    # Carichiamo il file solo se è diverso da quello già inviato in precedenza.
    # Senza questo controllo, ogni rerun di Streamlit richiamerebbe /upload
    # resettando lo stato del backend (trained=False) e nascondendo il pulsante Genera.
    if st.session_state.get("last_uploaded_file") != uploaded_file.name:
        with st.spinner("Caricamento e preprocessing in corso..."):
            response = api_post(
                "/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                timeout=60,
            )

        if response:
            data = response.json()
            st.session_state["last_uploaded_file"] = uploaded_file.name
            st.session_state["upload_data"]        = data

            uploaded_file.seek(0)
            st.session_state["real_df"]  = pd.read_csv(uploaded_file)
            st.session_state["uploaded"] = True

    if st.session_state.get("uploaded"):
        data = st.session_state["upload_data"]
        st.success(f"Dataset caricato: **{data['rows']}** righe, **{data['columns']}** colonne → **{data['input_dim']}** feature dopo preprocessing.")

        with st.expander("👁️ Anteprima dataset originale", expanded=False):
            st.dataframe(st.session_state["real_df"].head(10), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Righe", data["rows"])
            c2.metric("Colonne originali", data["columns"])
            c3.metric("Feature dopo preprocessing", data["input_dim"])

st.divider()


# =============================================================================
# SEZIONE 2 — TRAINING
# =============================================================================

st.header("2️⃣ Addestra il VAE")

status = api_get("/status")

if status:
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset caricato",     "✅ Sì" if status["has_data"]      else "❌ No")
    c2.metric("Training completato",  "✅ Sì" if status["trained"]       else "❌ No")
    c3.metric("Dati sintetici pronti","✅ Sì" if status["has_synthetic"] else "❌ No")

    if status["training"]:
        # Polling automatico ogni 3 secondi fino al completamento del training.
        st.info("⏳ Training in corso... attendere.")
        progress_bar = st.progress(0)
        status_text  = st.empty()
        iteration    = 0

        while True:
            time.sleep(3)
            current = api_get("/status")
            if current is None or not current["training"]:
                break
            iteration += 1
            progress_bar.progress(min(iteration * 5, 95))
            status_text.text(f"Training in corso... ({iteration * 3}s)")

        progress_bar.progress(100)
        status_text.text("Completato!")
        st.success("✅ VAE addestrato con successo!")
        st.rerun()

    elif not status["trained"]:
        if status["has_data"]:
            if st.button("🚀 Avvia Training", type="primary", use_container_width=True):
                response = api_post("/train", json={
                    "n_epochs":   int(n_epochs),
                    "latent_dim": int(latent_dim),
                    "hidden_dim": int(hidden_dim),
                    "beta":       float(beta),
                })
                if response:
                    st.success("Training avviato in background.")
                    time.sleep(1)
                    st.rerun()
        else:
            st.warning("Prima carica un dataset nella sezione 1.")

    else:
        st.success("✅ VAE già addestrato.")
        loss_plot = Path("outputs/training_loss.png")
        if loss_plot.exists():
            with st.expander("📈 Grafico di convergenza", expanded=False):
                st.image(str(loss_plot), use_container_width=True)

st.divider()


# =============================================================================
# SEZIONE 3 — GENERAZIONE E VALIDAZIONE
# =============================================================================

st.header("3️⃣ Genera e Valida i Dati Sintetici")

if status and status["trained"]:
    if st.button("⚡ Genera Dataset Sintetico", type="primary", use_container_width=True):
        with st.spinner(f"Generazione di {n_samples} campioni e validazione..."):
            response = api_post("/generate", json={"n_samples": int(n_samples)}, timeout=120)

        if response:
            result = response.json()
            st.success(f"Dataset sintetico generato: **{result['n_samples']}** record.")

            pc           = result["privacy_check"]
            privacy_icon = "✅" if pc["privacy_ok"] else "⚠️"

            with st.expander(f"{privacy_icon} Risultati Privacy Check", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Record a rischio", f"{pc['n_at_risk']} ({pc['pct_at_risk']}%)")
                c2.metric("Distanza minima",  pc["min_distance"])
                c3.metric("Distanza media",   pc["mean_min_distance"])
                if pc["privacy_ok"]:
                    st.success("Nessun record sintetico identico ai dati reali.")
                else:
                    st.warning(f"{pc['n_at_risk']} record troppo vicini ai dati reali.")
            st.rerun()

    # Grafici di validazione
    corr_plot = Path("outputs/correlation_comparison.png")
    dist_plot = Path("outputs/distribution_comparison.png")

    if corr_plot.exists() or dist_plot.exists():
        st.subheader("📊 Validazione Qualità")
        tab1, tab2, tab3 = st.tabs(["Correlazioni", "Distribuzioni", "Tabella Comparativa"])

        with tab1:
            if corr_plot.exists():
                st.image(str(corr_plot), use_container_width=True)

        with tab2:
            if dist_plot.exists():
                st.image(str(dist_plot), use_container_width=True)

        with tab3:
            synt_csv = Path("outputs/synthetic_data.csv")
            if synt_csv.exists() and "real_df" in st.session_state:
                real_df  = st.session_state["real_df"]
                synt_df  = pd.read_csv(synt_csv)
                num_cols = real_df.select_dtypes(include="number").columns.tolist()

                col_r, col_s = st.columns(2)
                with col_r:
                    st.caption("🔵 Dati REALI")
                    st.dataframe(real_df[num_cols].describe().round(3), use_container_width=True)
                with col_s:
                    st.caption("🟠 Dati SINTETICI")
                    synt_cols = [c for c in num_cols if c in synt_df.columns]
                    st.dataframe(synt_df[synt_cols].describe().round(3), use_container_width=True)

    # Download
    synt_csv = Path("outputs/synthetic_data.csv")
    if synt_csv.exists():
        st.subheader("⬇️ Scarica il Dataset Sintetico")
        with open(synt_csv, "rb") as f:
            st.download_button(
                label="📥 Scarica synthetic_data.csv",
                data=f,
                file_name="synthetic_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

elif status:
    st.warning("Prima completa il training nella sezione 2.")

st.divider()


# =============================================================================
# SEZIONE 4 — CHATBOT LLM
# =============================================================================

st.header("4️⃣ Natural Language Analyst 🤖")
st.caption("Fai domande in linguaggio naturale sul dataset sintetico.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Mostriamo la storia della conversazione.
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg.get("image"):
            st.image(msg["image"], use_container_width=True)
        if msg.get("code"):
            with st.expander("🔍 Codice Python generato"):
                st.code(msg["code"], language="python")
        if msg.get("content"):
            st.write(msg["content"])

synt_ready = Path("outputs/synthetic_data.csv").exists()

if not synt_ready:
    st.info("Il chatbot sarà disponibile dopo aver generato il dataset sintetico.")
else:
    user_input = st.chat_input("Es: 'Mostrami la distribuzione dell\\'età'")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analisi in corso..."):
                response = api_post("/chat", json={"question": user_input}, timeout=60)

            if response:
                import base64
                data     = response.json()
                code     = data.get("code", "")
                text_out = data.get("output", "")
                image_b64 = data.get("image")

                if image_b64:
                    image_bytes = base64.b64decode(image_b64)
                    st.image(image_bytes, use_container_width=True)
                elif text_out:
                    st.text(text_out)

                if code:
                    with st.expander("🔍 Codice Python generato"):
                        st.code(code, language="python")

                st.session_state["messages"].append({
                    "role":    "assistant",
                    "image":   base64.b64decode(image_b64) if image_b64 else None,
                    "code":    code,
                    "content": text_out,
                })

            else:
                msg = "Impossibile ottenere una risposta dal backend."
                st.error(msg)
                st.session_state["messages"].append({"role": "assistant", "content": msg})

        st.rerun()
