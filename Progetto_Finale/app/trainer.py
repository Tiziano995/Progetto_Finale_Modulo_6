import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from app.vae_model import VAE, vae_loss

# =============================================================================
# TRAINER — Ciclo di addestramento del VAE
# =============================================================================
# Questo modulo gestisce il training del VAE: ad ogni iterazione la rete
# elabora un batch di dati, calcola l'errore (loss) e aggiorna i propri pesi
# per migliorare progressivamente la qualità delle ricostruzioni.
#
# Il processo si ripete per un numero configurabile di epoche (passaggi
# completi sull'intero dataset) fino alla convergenza.
# =============================================================================


# =============================================================================
# IPERPARAMETRI
# =============================================================================
# Raccogliamo tutti i parametri qui per facilitarne la modifica.

BATCH_SIZE    = 64    # Numero di campioni elaborati per ogni aggiornamento dei pesi
NUM_EPOCHS    = 100   # Numero di passaggi completi sul dataset
LEARNING_RATE = 1e-3  # Velocità di aggiornamento dei pesi (Adam optimizer)
LATENT_DIM    = 16    # Dimensione dello spazio latente del VAE
HIDDEN_DIM    = 256   # Larghezza dei layer nascosti
BETA          = 1.0   # Peso della KL Divergence nella loss (1.0 = VAE standard)


# =============================================================================
# PREPARAZIONE DEI DATI
# =============================================================================

def build_dataloader(tensor: torch.Tensor) -> DataLoader:
    # Crea un DataLoader che fornisce i dati al VAE in mini-batch durante il training.
    # shuffle=True mescola i dati ad ogni epoca per evitare che la rete
    # impari l'ordine dei campioni invece delle loro caratteristiche.
    # drop_last=True scarta l'ultimo batch se incompleto (evita problemi con BatchNorm).
    dataset = TensorDataset(tensor)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"[Trainer] DataLoader: {len(dataset)} campioni, {len(loader)} batch per epoca")
    return loader


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(tensor: torch.Tensor, save_path: str = "models/vae.pth") -> VAE:
    # Addestra il VAE sul tensore fornito dal DataEngine e salva i pesi su disco.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trainer] Device: {device}")

    loader    = build_dataloader(tensor)
    input_dim = tensor.shape[1]

    vae = VAE(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
    print(f"[Trainer] VAE: input={input_dim}, hidden={HIDDEN_DIM}, latent={LATENT_DIM}")

    # Adam è l'ottimizzatore standard per le reti profonde:
    # adatta automaticamente il learning rate per ogni parametro.
    optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)

    history = {"total": [], "recon": [], "kl": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        vae.train()
        epoch_total = epoch_recon = epoch_kl = 0.0

        for (batch,) in loader:
            batch = batch.to(device)

            # Forward: il VAE elabora il batch e produce la ricostruzione.
            x_hat, mu, log_var = vae(batch)

            # Calcolo della loss: ricostruzione + KL Divergence.
            loss, recon_loss, kl_loss = vae_loss(batch, x_hat, mu, log_var, beta=BETA)

            # Backward: calcolo dei gradienti e aggiornamento dei pesi.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl    += kl_loss.item()

        n = len(loader.dataset)
        avg_total = epoch_total / n
        avg_recon = epoch_recon / n
        avg_kl    = epoch_kl    / n

        history["total"].append(avg_total)
        history["recon"].append(avg_recon)
        history["kl"].append(avg_kl)

        if epoch % 10 == 0 or epoch == 1 or epoch == NUM_EPOCHS:
            print(f"Epoca [{epoch:>3}/{NUM_EPOCHS}] | "
                  f"Loss: {avg_total:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

    # Salviamo i pesi del modello insieme agli iperparametri usati,
    # così possiamo ricaricare il modello in futuro senza dover ricordare i parametri.
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": vae.state_dict(),
        "input_dim":        input_dim,
        "hidden_dim":       HIDDEN_DIM,
        "latent_dim":       LATENT_DIM,
    }, save_path)
    print(f"[Trainer] Modello salvato in: {save_path}")

    _save_loss_plot(history)
    return vae


# =============================================================================
# CARICAMENTO DEL MODELLO
# =============================================================================

def load_vae(path: str = "models/vae.pth", device: torch.device = None) -> VAE:
    # Carica un VAE precedentemente addestrato, ricreando l'architettura
    # con gli stessi iperparametri e ripristinando i pesi salvati.
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)
    vae = VAE(
        input_dim  = checkpoint["input_dim"],
        hidden_dim = checkpoint["hidden_dim"],
        latent_dim = checkpoint["latent_dim"],
    ).to(device)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()
    print(f"[Trainer] VAE caricato da: {path}")
    return vae


# =============================================================================
# GRAFICO DELLA CONVERGENZA
# =============================================================================

def _save_loss_plot(history: dict, path: str = "outputs/training_loss.png"):
    # Salva un grafico con l'andamento delle tre loss durante il training.
    # La loss totale e la Reconstruction Loss devono scendere progressivamente;
    # la KL Divergence deve stabilizzarsi su un valore basso.
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels  = ["Loss Totale", "Reconstruction Loss (MSE)", "KL Divergence"]
    keys    = ["total", "recon", "kl"]
    colors  = ["steelblue", "darkorange", "seagreen"]

    for ax, key, label, color in zip(axes, keys, labels, colors):
        ax.plot(history[key], color=color)
        ax.set_title(label)
        ax.set_xlabel("Epoca")
        ax.grid(alpha=0.3)

    plt.suptitle("Convergenza del Training VAE", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Trainer] Grafico loss salvato in: {path}")
