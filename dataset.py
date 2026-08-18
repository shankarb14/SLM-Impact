

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import config as C

TEXT_COLUMN = {"codemixed": "comment", "translated": "translated"}
LABEL_COLUMN = {"binary": "binary_label", "multiclass": "fine_label"}
RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}


class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.enc = tokenizer(list(texts), truncation=True, padding="max_length",
                             max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "label": self.labels[i],
        }


def load_corpus(path=None, task: str = "binary") -> pd.DataFrame:
    path = path or (C.DATA / "normalized.csv")
    df = pd.read_csv(path)

    required = {"comment", "translated", "binary_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    if task == "multiclass":
        if "fine_label" not in df.columns:
            raise ValueError("fine-grained task requires a 'fine_label' column")
        df = df[df.binary_label == "Hate"].dropna(subset=["fine_label"]).reset_index(drop=True)
    return df


def split(df: pd.DataFrame, label_col: str, seed: int = C.SEED):
    """70:10:20, stratified, in two steps so val and test are both stratified."""
    idx, y = df.index.to_numpy(), df[label_col].to_numpy()
    train_i, hold_i, _, hold_y = train_test_split(
        idx, y, test_size=RATIOS["val"] + RATIOS["test"], stratify=y, random_state=seed)
    val_frac = RATIOS["val"] / (RATIOS["val"] + RATIOS["test"])
    val_i, test_i = train_test_split(
        hold_i, test_size=1 - val_frac, stratify=hold_y, random_state=seed)
    return df.loc[train_i], df.loc[val_i], df.loc[test_i]


def build_loaders(task: str, modality: str, tokenizer,
                  corpus_path=None, batch_size: int | None = None, seed: int = C.SEED):
    """Return (loaders, labels). `modality` selects code-mixed or translated input."""
    if task not in C.LABELS:
        raise ValueError(f"task must be one of {list(C.LABELS)}")
    if modality not in TEXT_COLUMN:
        raise ValueError(f"modality must be one of {list(TEXT_COLUMN)}")

    df = load_corpus(corpus_path, task)
    labels = C.LABELS[task]
    l2i = {l: i for i, l in enumerate(labels)}
    text_col, label_col = TEXT_COLUMN[modality], LABEL_COLUMN[task]
    bs = batch_size or C.HP["batch_size"]

    parts = dict(zip(("train", "val", "test"), split(df, label_col, seed)))
    loaders = {}
    for name, part in parts.items():
        ds = CommentDataset(part[text_col].astype(str).tolist(),
                            [l2i[v] for v in part[label_col]],
                            tokenizer, C.HP["max_seq_len"])
        loaders[name] = DataLoader(ds, batch_size=bs, shuffle=(name == "train"),
                                   num_workers=2)
        print(f"  {name:5s} n={len(ds):5d}  " +
              "  ".join(f"{k}={v}" for k, v in part[label_col].value_counts().items()))
    return loaders, labels
