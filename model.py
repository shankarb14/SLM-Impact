

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

import config as C


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over real tokens only, so padding does not dilute the representation."""
    m = mask.unsqueeze(-1).float()
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


class HateSpeechClassifier(nn.Module):
   

    def __init__(self, encoder_key: str, n_classes: int,
                 use_bilstm: bool = True, hp: dict | None = None):
        super().__init__()
        if encoder_key not in C.ENCODERS:
            raise ValueError(f"unknown encoder '{encoder_key}'; choose from {list(C.ENCODERS)}")
        hp = hp or C.HP

        self.encoder_key = encoder_key
        self.encoder = AutoModel.from_pretrained(C.ENCODERS[encoder_key])
        if hp["freeze_encoder"]:
            for p in self.encoder.parameters():
                p.requires_grad = False

        d = self.encoder.config.hidden_size
        self.use_bilstm = use_bilstm

        if use_bilstm:
            self.bilstm = nn.LSTM(input_size=d, hidden_size=hp["bilstm_hidden"],
                                  num_layers=1, batch_first=True, bidirectional=True)
            feat_dim = hp["bilstm_hidden"] * 2      # forward state ; backward state
        else:
            self.bilstm = None
            feat_dim = d

        self.dropout = nn.Dropout(hp["dropout"])
        self.dense = nn.Linear(feat_dim, hp["dense_hidden"])
        self.relu = nn.ReLU()

       
        self.head = nn.Linear(hp["dense_hidden"], 1 if n_classes == 2 else n_classes)
        self.n_classes = n_classes

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if not any(p.requires_grad for p in self.encoder.parameters()):
            with torch.no_grad():
                hidden = self.encoder(input_ids=input_ids,
                                      attention_mask=attention_mask).last_hidden_state
        else:
            hidden = self.encoder(input_ids=input_ids,
                                  attention_mask=attention_mask).last_hidden_state

        if self.use_bilstm:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                hidden, lengths, batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.bilstm(packed)
            feat = torch.cat([h_n[0], h_n[1]], dim=-1)   # [h_fwd ; h_bwd]
        else:
            feat = masked_mean(hidden, attention_mask)

        z = self.relu(self.dense(self.dropout(feat)))
        logits = self.head(z)
        return logits.squeeze(-1) if self.n_classes == 2 else logits

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Return (predicted indices, probabilities)."""
        self.eval()
        logits = self(input_ids, attention_mask)
        if self.n_classes == 2:
            p_hate = torch.sigmoid(logits)
            preds = (p_hate >= 0.5).long()
            probs = torch.stack([p_hate, 1 - p_hate], dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(-1)
        return preds, probs

    def train(self, mode: bool = True):
        """Keep a frozen encoder in eval mode so its dropout stays disabled."""
        super().train(mode)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        return self


def get_tokenizer(encoder_key: str):
    return AutoTokenizer.from_pretrained(C.ENCODERS[encoder_key])


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
