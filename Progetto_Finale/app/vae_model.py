import torch
import torch.nn as nn

# =============================================================================
# VAE MODEL — Variational Autoencoder
# =============================================================================
# Un Variational Autoencoder è una rete neurale che impara a "capire" i dati
# comprimendoli in uno spazio ridotto (spazio latente) e poi ricostruendoli.
#
# È composto da tre parti:
#   - Encoder: comprime i dati in una distribuzione gaussiana (media + varianza).
#   - Reparameterization Trick: campiona un punto dalla distribuzione in modo
#     che il training con backpropagation possa funzionare correttamente.
#   - Decoder: ricostruisce i dati originali a partire dal punto campionato.
#
# Per generare dati sintetici nuovi, si campionano punti casuali direttamente
# dallo spazio latente (senza passare per l'Encoder) e si passano al Decoder.
# =============================================================================


class Encoder(nn.Module):
    # Comprime un record del dataset in due vettori:
    #   - mu (media): il centro della distribuzione nello spazio latente.
    #   - log_var (logaritmo della varianza): quanto è "larga" la distribuzione.
    # Usare il logaritmo della varianza evita valori negativi durante il calcolo.

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        # Rete principale: due layer che comprimono progressivamente i dati.
        # ReLU è la funzione di attivazione standard per layer nascosti.
        # BatchNorm1d stabilizza il training normalizzando le attivazioni.
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Due teste separate: una produce mu, l'altra produce log_var.
        self.fc_mu      = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x: torch.Tensor):
        h  = self.backbone(x)
        mu = self.fc_mu(h)
        lv = self.fc_log_var(h)
        return mu, lv


# =============================================================================
# REPARAMETERIZATION TRICK
# =============================================================================

def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    # Problema: il campionamento casuale non è differenziabile,
    # quindi il gradiente non può attraversarlo durante il training.
    #
    # Soluzione: invece di campionare z direttamente da N(mu, sigma²),
    # campioniamo epsilon da N(0, 1) e calcoliamo z = mu + sigma * epsilon.
    # La casualità è "spostata" su epsilon (fisso), mentre mu e log_var
    # restano differenziabili e i loro gradienti possono essere calcolati.

    sigma   = torch.exp(0.5 * log_var)
    epsilon = torch.randn_like(sigma)
    return mu + sigma * epsilon


# =============================================================================
# DECODER
# =============================================================================

class Decoder(nn.Module):
    # Ricostruisce un record del dataset a partire da un punto nello spazio latente.
    # In fase di generazione, questo punto viene campionato casualmente da N(0, 1).

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        # Architettura speculare all'Encoder: parte dallo spazio latente
        # e aumenta progressivamente la dimensione fino all'output originale.
        # Sigmoid finale: porta i valori in (0, 1), coerente con MinMaxScaler.
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# =============================================================================
# VAE — Assemblaggio completo
# =============================================================================

class VAE(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 32):
        super().__init__()
        self.encoder   = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder   = Decoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim
        self.input_dim  = input_dim

    def forward(self, x: torch.Tensor):
        # Encoder → campionamento → Decoder.
        # Restituiamo anche mu e log_var perché servono per calcolare la loss.
        mu, log_var = self.encoder(x)
        z           = reparameterize(mu, log_var)
        x_hat       = self.decoder(z)
        return x_hat, mu, log_var

    def generate(self, n_samples: int, device: torch.device) -> torch.Tensor:
        # Generazione di dati sintetici: campiona punti casuali dallo spazio latente
        # e li passa al Decoder. L'Encoder non viene usato in questa fase.
        self.eval()
        with torch.no_grad():
            z     = torch.randn(n_samples, self.latent_dim, device=device)
            x_hat = self.decoder(z)
        return x_hat


# =============================================================================
# LOSS FUNCTION
# =============================================================================

def vae_loss(x: torch.Tensor, x_hat: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor,
             beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # La loss del VAE è la somma di due termini:
    #
    # 1. Reconstruction Loss (MSE): misura quanto la ricostruzione del Decoder
    #    si avvicina all'input originale. Più è bassa, meglio il modello ricostruisce.
    #
    # 2. KL Divergence: misura quanto la distribuzione latente si discosta
    #    dalla gaussiana standard N(0, 1). Questo "regolarizza" lo spazio latente,
    #    rendendolo continuo e adatto alla generazione di nuovi dati.
    #
    # Il parametro beta bilancia i due termini (beta=1 è il VAE standard).

    recon_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
    kl_div     = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = recon_loss + beta * kl_div

    return total_loss, recon_loss, kl_div
