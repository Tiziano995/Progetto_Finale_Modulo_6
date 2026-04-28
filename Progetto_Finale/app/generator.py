import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cdist

from app.vae_model import VAE
from app.data_engine import DataEngine

# =============================================================================
# GENERATOR — Generazione e validazione dei dati sintetici
# =============================================================================
# Questo modulo si occupa di due cose:
#
# 1. GENERAZIONE: usa il Decoder del VAE addestrato per produrre nuovi record
#    sintetici campionando punti casuali dallo spazio latente.
#
# 2. VALIDAZIONE: verifica che i dati sintetici siano di buona qualità:
#    - Le correlazioni tra variabili sono simili a quelle reali?
#    - Le distribuzioni dei valori hanno la stessa forma?
#    - Nessun dato sintetico è identico (o quasi) a quelli originali?
# =============================================================================


# =============================================================================
# GENERAZIONE
# =============================================================================

def generate_synthetic(
    vae: VAE,
    engine: DataEngine,
    n_samples: int = 1000,
    device: torch.device = None,
    save_csv: str = "outputs/synthetic_data.csv",
) -> pd.DataFrame:
    # Genera n_samples record sintetici e li converte in un DataFrame leggibile.

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Generator] Generazione di {n_samples} campioni sintetici...")

    # Il VAE campiona punti casuali da N(0, 1) e li passa al Decoder.
    synthetic_tensor = vae.generate(n_samples, device)

    # Il DataEngine converte il tensore nei valori originali (inverse transform).
    synthetic_df = engine.postprocess(synthetic_tensor.cpu())

    Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(save_csv, index=False)
    print(f"[Generator] Dataset sintetico salvato in: {save_csv}")

    return synthetic_df


# =============================================================================
# VALIDAZIONE — Confronto correlazioni
# =============================================================================

def plot_correlation_comparison(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    save_path: str = "outputs/correlation_comparison.png",
):
    # Mostra affiancate le matrici di correlazione del dataset reale e sintetico.
    # Se il VAE ha imparato bene, le due heatmap devono essere visivamente simili:
    # stessi colori nelle stesse posizioni indicano che le relazioni tra variabili
    # sono state preservate.

    num_cols  = real_df.select_dtypes(include=[np.number]).columns.tolist()
    real_corr = real_df[num_cols].corr()
    synt_corr = synthetic_df[num_cols].corr()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, corr, title in zip(
        axes,
        [real_corr, synt_corr],
        ["Correlazione — Dati REALI", "Correlazione — Dati SINTETICI"]
    ):
        sns.heatmap(corr, ax=ax, annot=True, fmt=".2f",
                    cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
        ax.set_title(title, fontsize=13, fontweight="bold")

    plt.suptitle("Confronto Matrici di Correlazione", fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Generator] Heatmap correlazioni salvata in: {save_path}")


# =============================================================================
# VALIDAZIONE — Confronto distribuzioni
# =============================================================================

def plot_distribution_comparison(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    n_cols: int = 4,
    save_path: str = "outputs/distribution_comparison.png",
):
    # Confronta gli istogrammi di ogni variabile numerica tra dati reali e sintetici.
    # Le barre blu rappresentano i dati reali, quelle arancioni i dati sintetici.
    # Le forme devono essere simili: se coincidono il VAE ha imparato bene la distribuzione.

    num_cols   = real_df.select_dtypes(include=[np.number]).columns.tolist()
    n_features = len(num_cols)
    n_rows     = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        ax.hist(real_df[col].dropna(),      bins=30, alpha=0.6, color="steelblue",  label="Reale",     density=True)
        ax.hist(synthetic_df[col].dropna(), bins=30, alpha=0.6, color="darkorange", label="Sintetico", density=True)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xlabel("Valore")
        ax.set_ylabel("Densità")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Confronto Distribuzioni: Dati Reali vs Sintetici", fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Generator] Confronto distribuzioni salvato in: {save_path}")


# =============================================================================
# VALIDAZIONE — Privacy Check
# =============================================================================

def privacy_check(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    threshold: float = 0.01,
) -> dict:
    # Verifica che nessun dato sintetico sia troppo simile a uno reale.
    # Per ogni record sintetico calcoliamo la distanza dal record reale più vicino:
    # se questa distanza è inferiore alla soglia, il record è "a rischio"
    # perché potrebbe rivelare informazioni sui dati originali.

    print("[Generator] Esecuzione Privacy Check...")

    num_cols    = real_df.select_dtypes(include=[np.number]).columns.tolist()
    common_cols = [c for c in num_cols if c in synthetic_df.columns]
    real_vals   = real_df[common_cols].dropna().values
    synt_vals   = synthetic_df[common_cols].dropna().values

    # Calcoliamo le distanze tra ogni record sintetico e tutti i record reali.
    dist_matrix  = cdist(synt_vals, real_vals, metric="euclidean")
    min_distances = dist_matrix.min(axis=1)

    n_at_risk   = int((min_distances < threshold).sum())
    pct_at_risk = n_at_risk / len(min_distances) * 100

    results = {
        "n_synthetic":       len(min_distances),
        "n_at_risk":         n_at_risk,
        "pct_at_risk":       round(pct_at_risk, 2),
        "min_distance":      round(float(min_distances.min()), 6),
        "mean_min_distance": round(float(min_distances.mean()), 6),
        "threshold":         threshold,
        "privacy_ok":        n_at_risk == 0,
    }

    print(f"[Privacy Check] Campioni a rischio: {n_at_risk} ({pct_at_risk:.2f}%)")
    print(f"[Privacy Check] Distanza minima: {results['min_distance']}")

    if results["privacy_ok"]:
        print("[Privacy Check] SUPERATO: nessun record troppo simile ai dati reali.")
    else:
        print(f"[Privacy Check] ATTENZIONE: {n_at_risk} record troppo simili ai dati reali.")

    return results


# =============================================================================
# PIPELINE COMPLETA DI VALIDAZIONE
# =============================================================================

def validate_all(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> dict:
    # Esegue tutte le validazioni in sequenza e restituisce i risultati del privacy check.
    print("\n[Generator] === INIZIO VALIDAZIONE ===")
    plot_correlation_comparison(real_df, synthetic_df)
    plot_distribution_comparison(real_df, synthetic_df)
    privacy_results = privacy_check(real_df, synthetic_df)
    print("[Generator] === FINE VALIDAZIONE ===\n")
    return privacy_results
