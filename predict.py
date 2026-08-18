

import argparse
import html
import re
import unicodedata
from pathlib import Path

import torch

import config as C
from model import HateSpeechClassifier, get_tokenizer

URL = re.compile(r"https?://\S+|www\.\S+")
HANDLE = re.compile(r"@\w+")
WHITESPACE = re.compile(r"\s+")


def clean(text: str) -> str:
   
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = URL.sub(" ", text)
    text = HANDLE.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


class HateSpeechPredictor:
    

    def __init__(self, checkpoint: Path, en_vocab: Path | None = None):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.labels = ckpt["labels"]
        self.task = ckpt["task"]
        self.modality = ckpt["modality"]

        self.tokenizer = get_tokenizer(ckpt["encoder"])
        self.model = HateSpeechClassifier(ckpt["encoder"], len(self.labels),
                                          use_bilstm=ckpt["use_bilstm"], hp=ckpt["hp"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(C.device()).eval()

        self._pipeline = None
        self._en_vocab = en_vocab
        print(f"loaded {checkpoint.name}: {ckpt['encoder']} | {self.task} | {self.modality}")

    @property
    def pipeline(self):
        if self._pipeline is None:
            from normalization import NormalizationPipeline
            print("loading IndicXlit + IndicTrans2 (first call only)...")
            self._pipeline = NormalizationPipeline(self._en_vocab)
        return self._pipeline

    def _prepare(self, texts: list[str]) -> list[str]:
        texts = [clean(t) for t in texts]                       # step 1
        if self.modality == "translated":                       # steps 2-5
            texts = [self.pipeline(t) for t in texts]
        return texts

    @torch.no_grad()
    def __call__(self, texts: str | list[str]) -> list[dict]:
        single = isinstance(texts, str)
        raw = [texts] if single else list(texts)
        prepared = self._prepare(raw)

        enc = self.tokenizer(prepared, truncation=True, padding="max_length",
                             max_length=C.HP["max_seq_len"], return_tensors="pt").to(C.device())
        preds, probs = self.model.predict(enc["input_ids"], enc["attention_mask"])

        out = []
        for i, original in enumerate(raw):
            dist = probs[i].cpu().tolist()
            out.append(dict(
                text=original,
                model_input=prepared[i],
                label=self.labels[preds[i].item()],
                confidence=round(max(dist), 4),
                distribution={l: round(p, 4) for l, p in zip(self.labels, dist)},
            ))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--text", help="a single comment")
    ap.add_argument("--file", type=Path, help="one comment per line")
    ap.add_argument("--en_vocab", type=Path, default=None)
    a = ap.parse_args()

    if not a.text and not a.file:
        ap.error("provide --text or --file")

    predictor = HateSpeechPredictor(a.checkpoint, a.en_vocab)
    texts = [a.text] if a.text else [
        l.strip() for l in a.file.read_text(encoding="utf-8").splitlines() if l.strip()]

    for r in predictor(texts):
        print("\n" + "-" * 66)
        print(f"input      : {r['text']}")
        if r["model_input"] != r["text"]:
            print(f"normalized : {r['model_input']}")
        print(f"prediction : {r['label']}  (confidence {r['confidence']:.4f})")
        ranked = sorted(r["distribution"].items(), key=lambda kv: -kv[1])
        print("             " + "  ".join(f"{l}={p:.3f}" for l, p in ranked))


if __name__ == "__main__":
    main()
