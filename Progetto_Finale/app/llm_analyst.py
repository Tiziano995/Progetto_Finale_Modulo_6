import io
import traceback
import contextlib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Backend "Agg": genera grafici in memoria senza aprire finestre grafiche.
# Necessario in ambienti server come FastAPI.
matplotlib.use("Agg")

# Carichiamo la API key dal file .env (evita di scrivere credenziali nel codice).
load_dotenv()

# =============================================================================
# LLM ANALYST — Analisi in linguaggio naturale
# =============================================================================
# Questo modulo permette all'utente di fare domande sul dataset sintetico
# in linguaggio naturale e ottenere grafici o statistiche come risposta.
#
# Flusso di lavoro:
#   1. L'utente scrive una domanda (es. "Mostrami la distribuzione dell'età").
#   2. La domanda viene inviata a GPT-4o-mini con un contesto che descrive
#      la struttura del dataset (nomi e tipi delle colonne).
#   3. L'LLM risponde con codice Python (Pandas + Matplotlib).
#   4. Il codice viene eseguito in un ambiente controllato (sandbox).
#   5. Se il codice produce un grafico, viene restituito come immagine PNG.
#      Se produce testo, viene restituito come stringa.
# =============================================================================


# =============================================================================
# CLIENT OPENAI
# =============================================================================

def get_client() -> OpenAI:
    # Crea il client OpenAI. La API key viene letta automaticamente
    # dalla variabile d'ambiente OPENAI_API_KEY caricata dal file .env.
    return OpenAI()


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

def build_system_prompt(df: pd.DataFrame) -> str:
    # Costruisce il messaggio di istruzioni che viene inviato all'LLM
    # prima di ogni domanda. Contiene la struttura del dataset
    # (nomi colonne, tipi, esempi di valori) e le regole di risposta.

    col_descriptions = []
    for col in df.columns:
        dtype    = str(df[col].dtype)
        examples = df[col].dropna().unique()[:3].tolist()
        col_descriptions.append(f"  - '{col}' (tipo: {dtype}, es: {examples})")

    cols_text = "\n".join(col_descriptions)

    return f"""Sei un analista dati esperto. Hai accesso a un DataFrame Pandas chiamato `df`
che contiene dati sintetici e anonimi generati da un VAE.

Il DataFrame ha {len(df)} righe e le seguenti colonne:
{cols_text}

REGOLE:
1. Rispondi SEMPRE e SOLO con codice Python eseguibile, senza spiegazioni.
2. Il DataFrame `df` è già disponibile: non caricarlo da file.
3. Le seguenti librerie sono già disponibili, NON importarle:
   - `plt` (matplotlib.pyplot)
   - `pd` (pandas)
   - `np` (numpy)
4. NON scrivere MAI righe che iniziano con `import` o `from`.
5. Chiudi sempre con `plt.tight_layout()`, ma NON chiamare `plt.show()`.
6. Per risultati testuali usa `print()`.
"""


# =============================================================================
# CHIAMATA ALL'API OPENAI
# =============================================================================

def ask_llm(question: str, df: pd.DataFrame, chat_history: list,
            model: str = "gpt-4o-mini") -> str:
    # Invia la domanda all'API OpenAI insieme alla storia della conversazione
    # e restituisce il codice Python generato dall'LLM.
    # temperature=0.2: risposte più precise e deterministiche (ideale per il codice).

    client   = get_client()
    messages = [
        {"role": "system", "content": build_system_prompt(df)},
        *chat_history,
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )

    raw_code = response.choices[0].message.content
    return _clean_code(raw_code)


def _clean_code(raw: str) -> str:
    # Rimuove i delimitatori markdown (```python ... ```) che l'LLM aggiunge
    # attorno al codice, rendendolo non eseguibile direttamente.
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw   = "\n".join(lines[1:-1])

    # Rimuoviamo le righe di import: plt, pd e np sono già nel namespace sandbox.
    # Se il LLM li importa comunque, la sandbox blocca l'esecuzione con ImportError.
    filtered = [
        line for line in raw.splitlines()
        if not line.strip().startswith("import ") and not line.strip().startswith("from ")
    ]
    return "\n".join(filtered).strip()


# =============================================================================
# ESECUZIONE SICURA DEL CODICE
# =============================================================================

def execute_code(code: str, df: pd.DataFrame) -> tuple[str, bytes | None]:
    # Esegue il codice generato dall'LLM in un ambiente sandbox.
    # Il codice ha accesso solo a: df, pd, plt, np.
    # Moduli pericolosi come os, sys, subprocess non sono disponibili.
    #
    # Restituisce:
    #   - testo: tutto ciò che è stato stampato con print().
    #   - immagine: il grafico come bytes PNG, oppure None se non c'è grafico.

    import numpy as np

    safe_builtins = {
        "print": print, "len": len, "range": range, "enumerate": enumerate,
        "zip": zip, "list": list, "dict": dict, "tuple": tuple, "set": set,
        "str": str, "int": int, "float": float, "bool": bool,
        "round": round, "abs": abs, "min": min, "max": max, "sum": sum,
        "sorted": sorted, "reversed": reversed, "isinstance": isinstance,
    }

    sandbox = {
        "__builtins__": safe_builtins,
        "df":  df.copy(),
        "pd":  pd,
        "plt": plt,
        "np":  np,
    }

    output_buffer = io.StringIO()
    image_bytes   = None

    try:
        plt.close("all")
        with contextlib.redirect_stdout(output_buffer):
            exec(code, sandbox)

        if plt.get_fignums():
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            image_bytes = buf.getvalue()
            plt.close("all")

    except Exception:
        return f"Errore nell'esecuzione del codice:\n{traceback.format_exc()}", None

    return output_buffer.getvalue().strip(), image_bytes


# =============================================================================
# PIPELINE COMPLETA
# =============================================================================

def analyze(question: str, df: pd.DataFrame, chat_history: list,
            ) -> tuple[str, str, bytes | None]:
    # Pipeline completa: domanda → codice LLM → esecuzione → risultato.
    # Restituisce: (codice_generato, output_testo, immagine_png).

    print(f"[LLM Analyst] Domanda: {question}")
    code = ask_llm(question, df, chat_history)
    print(f"[LLM Analyst] Codice generato:\n{code}")

    text_output, image_bytes = execute_code(code, df)
    return code, text_output, image_bytes
