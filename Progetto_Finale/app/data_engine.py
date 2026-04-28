import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# DATA ENGINE — Preparazione dei dati
# =============================================================================
# Questo modulo si occupa di tutto ciò che riguarda la pulizia e la trasformazione
# del dataset prima che venga passato al VAE.
#
# Flusso di lavoro:
#   1. Caricamento del CSV
#   2. Rilevamento automatico delle colonne (numeriche vs categoriche)
#   3. Gestione dei valori mancanti
#   4. Codifica delle variabili testuali (One-Hot Encoding)
#   5. Normalizzazione dei valori numerici in [0, 1]
#   6. Conversione in tensore PyTorch
# =============================================================================


class DataEngine:

    def __init__(self):
        # MinMaxScaler: porta tutti i valori numerici nell'intervallo [0, 1].
        # Necessario perché il VAE lavora meglio con valori in questa scala.
        self.scaler = MinMaxScaler()

        # SimpleImputer: sostituisce i valori mancanti (NaN) con la media della colonna.
        self.imputer = SimpleImputer(strategy="mean")

        self.categorical_columns = []   # Colonne testuali (es. "genere", "città")
        self.numerical_columns   = []   # Colonne numeriche (es. "età", "reddito")
        self.binary_columns      = []   # Colonne con solo valori 0/1 (es. "Outcome")
        self.feature_names       = []   # Nomi di tutte le colonne dopo il preprocessing
        self.original_columns    = []   # Nomi originali del CSV prima di qualsiasi modifica


    # -------------------------------------------------------------------------
    # CARICAMENTO
    # -------------------------------------------------------------------------

    def load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        self.original_columns = df.columns.tolist()
        print(f"[DataEngine] Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
        return df


    # -------------------------------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------------------------------

    def _detect_column_types(self, df: pd.DataFrame):
        # Classifica automaticamente ogni colonna in base al suo tipo di dato.
        self.categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.numerical_columns   = df.select_dtypes(include=[np.number]).columns.tolist()

        # Rileviamo le colonne binarie (solo valori 0 e 1): vanno arrotondate
        # in postprocess perché il VAE genera valori continui anche per queste.
        self.binary_columns = [
            col for col in self.numerical_columns
            if df[col].dropna().isin([0, 1]).all()
        ]

        print(f"[DataEngine] Numeriche: {self.numerical_columns}")
        print(f"[DataEngine] Categoriche: {self.categorical_columns}")
        print(f"[DataEngine] Binarie: {self.binary_columns}")

    def preprocess(self, df: pd.DataFrame) -> torch.Tensor:
        # Trasforma il DataFrame grezzo in un tensore PyTorch normalizzato.

        self._detect_column_types(df)

        # Sostituiamo i valori mancanti nelle colonne numeriche con la media.
        if self.numerical_columns:
            df[self.numerical_columns] = self.imputer.fit_transform(df[self.numerical_columns])

        # Per le colonne testuali usiamo il valore più frequente (moda).
        for col in self.categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])

        # One-Hot Encoding: converte ogni colonna testuale in colonne binarie (0/1).
        # Es: colonna "genere" → "genere_M" e "genere_F".
        if self.categorical_columns:
            df = pd.get_dummies(df, columns=self.categorical_columns, dtype=float)

        self.feature_names = df.columns.tolist()

        # Normalizziamo tutti i valori in [0, 1] e convertiamo in tensore PyTorch.
        scaled = self.scaler.fit_transform(df.values)
        tensor = torch.tensor(scaled, dtype=torch.float32)

        print(f"[DataEngine] Preprocessing completato. Dimensione input VAE: {tensor.shape[1]}")
        return tensor


    # -------------------------------------------------------------------------
    # POSTPROCESSING (da tensore sintetico a DataFrame leggibile)
    # -------------------------------------------------------------------------

    def postprocess(self, tensor: torch.Tensor) -> pd.DataFrame:
        # Converte il tensore prodotto dal VAE in un DataFrame con i valori originali.

        array = tensor.detach().numpy()
        array = np.clip(array, 0, 1)                        # Limitiamo i valori in [0, 1]
        restored = self.scaler.inverse_transform(array)     # Riportiamo alle scale originali
        df = pd.DataFrame(restored, columns=self.feature_names)

        # Ricostruiamo le colonne categoriche dal One-Hot Encoding:
        # prendiamo la categoria con il valore più alto per ogni riga.
        for cat_col in self.categorical_columns:
            ohe_cols = [c for c in self.feature_names if c.startswith(f"{cat_col}_")]
            if ohe_cols:
                df[cat_col] = df[ohe_cols].idxmax(axis=1).str.replace(f"{cat_col}_", "", regex=False)
                df.drop(columns=ohe_cols, inplace=True)

        # Per le colonne binarie non usiamo l'arrotondamento a 0.5 (che fallirebbe
        # se il VAE genera tutti valori < 0.5). Usiamo invece un campionamento
        # probabilistico: il valore sintetico normalizzato rappresenta la probabilità
        # di essere 1, quindi campioniamo da una distribuzione di Bernoulli.
        # Es: valore sintetico = 0.35 → 35% di probabilità di Outcome=1.
        for col in self.binary_columns:
            if col in df.columns:
                probs = df[col].clip(0, 1).values
                df[col] = (np.random.rand(len(df)) < probs).astype(int)

        return df


    # -------------------------------------------------------------------------
    # SALVATAGGIO E CARICAMENTO
    # -------------------------------------------------------------------------

    def save(self, path: str = "models/data_engine.pkl"):
        # Salviamo il DataEngine su disco (scaler + metadati colonne)
        # per poterlo riutilizzare in fase di generazione senza riaddestrarlo.
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[DataEngine] Salvato in: {path}")

    @staticmethod
    def load_engine(path: str = "models/data_engine.pkl") -> "DataEngine":
        with open(path, "rb") as f:
            engine = pickle.load(f)
        print(f"[DataEngine] Caricato da: {path}")
        return engine


    # -------------------------------------------------------------------------
    # PROPRIETÀ
    # -------------------------------------------------------------------------

    @property
    def input_dim(self) -> int:
        # Numero di feature dopo il preprocessing: serve al VAE per definire
        # la dimensione del layer di input e di output.
        return len(self.feature_names)
