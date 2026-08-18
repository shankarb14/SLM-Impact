

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"
for _p in (DATA, CKPT):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42


BINARY_LABELS = ["Hate", "Non-Hate"]
FINE_LABELS = ["Gender", "Political", "Religion", "Geo-political", "Violence", "Others"]
LABELS = {"binary": BINARY_LABELS, "multiclass": FINE_LABELS}


ENCODERS = {
    # standard-capacity encoders
    "BERT":       "bert-base-uncased",
    "mBERT":      "bert-base-multilingual-cased",
    "HateBERT":   "GroNLP/hateBERT",
    "XLM-R":      "xlm-roberta-base",
    "ALBERT":     "albert-base-v2",
    # distilled compact encoders
    "TinyBERT":   "huawei-noah/TinyBERT_General_4L_312D",
    "MobileBERT": "google/mobilebert-uncased",
    "DistilBERT": "distilbert-base-uncased",
}

# ---------------------------------------------------------------------------
# Hyperparameters (Table 4)
# ---------------------------------------------------------------------------
HP = dict(
    batch_size=16,
    lr=1e-3,                  # for the randomly initialised downstream layers
    adam_betas=(0.9, 0.999),
    max_epochs=10,
    early_stopping_patience=3,
    dropout=0.3,
    max_seq_len=128,
    bilstm_hidden=128,
    dense_hidden=128,
    freeze_encoder=True,      # only Bi-LSTM + dense + head are trained
)

# ---------------------------------------------------------------------------
# Normalization pipeline (Sect. III-C, Figures 1 and 2)
# ---------------------------------------------------------------------------
XLIT_MODEL = "ai4bharat/IndicXlit"
XLIT_LANG = "kn"
XLIT_BEAM = 4                 # beam search width used at decode time

NMT_MODEL = "ai4bharat/indictrans2-indic-en-1B"
NMT_SRC_TAG = "kan_Knda"
NMT_TGT_TAG = "eng_Latn"
NMT_BEAM = 5
NMT_MAX_LEN = 256


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = SEED):
    import os, random
    import numpy as np
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
