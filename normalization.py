

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config as C

# Tokens that are never transliterated: handles, hashtags, URLs, numerals,
# punctuation, emoji, and anything already in Kannada script.
NEUTRAL = re.compile(
    r"^(?:@\w+|#\w+|https?://\S+|[\d.,:%/+-]+|[^\w\s]+"
    r"|[\U0001F300-\U0001FAFF\u2600-\u27BF]+)$"
)
KANNADA_SCRIPT = re.compile(r"[\u0C80-\u0CFF]")
PLACEHOLDER = "\u2581{}\u2581"


class WordLID:
    """Tag each whitespace token as 'kn', 'en', or 'neutral'.

    Pass an explicit English word list for a real run; the NLTK word corpus is
    used as a fallback so the pipeline is runnable without extra resources.
    """

    def __init__(self, en_vocab_path: Path | None = None):
        self.en_vocab: set[str] = set()
        if en_vocab_path and en_vocab_path.exists():
            self.en_vocab = {w.strip().lower()
                             for w in en_vocab_path.read_text(encoding="utf-8").splitlines()
                             if w.strip()}
        else:
            try:
                from nltk.corpus import words as nltk_words
                self.en_vocab = {w.lower() for w in nltk_words.words()}
            except Exception:
                print("WARNING: no English vocabulary loaded — every alphabetic token "
                      "will be treated as Romanised Kannada.")

    def tag(self, token: str) -> str:
        if NEUTRAL.match(token) or KANNADA_SCRIPT.search(token):
            return "neutral"
        return "en" if token.lower() in self.en_vocab else "kn"


def mask(text: str, lid: WordLID) -> tuple[str, dict[str, str]]:
    """Replace non-Kannada tokens with placeholders before transliteration."""
    kept, pieces = {}, []
    for i, tok in enumerate(text.split()):
        if lid.tag(tok) == "kn":
            pieces.append(tok)
        else:
            ph = PLACEHOLDER.format(i)
            kept[ph] = tok
            pieces.append(ph)
    return " ".join(pieces), kept


def unmask(text: str, kept: dict[str, str]) -> str:
    for ph, tok in kept.items():
        text = text.replace(ph, tok)
    return text


# ---------------------------------------------------------------------------
# Stage 1 — transliteration
# ---------------------------------------------------------------------------
class Transliterator:
    def __init__(self, en_vocab: Path | None = None):
        from ai4bharat.transliteration import XlitEngine
        self.engine = XlitEngine(C.XLIT_LANG, beam_width=C.XLIT_BEAM, rescore=True)
        self.lid = WordLID(en_vocab)

    def __call__(self, text: str) -> str:
        masked, kept = mask(text, self.lid)
        out = []
        for tok in masked.split():
            if tok in kept:
                out.append(tok)
                continue
            cands = self.engine.translit_word(tok, topk=1).get(C.XLIT_LANG, [])
            out.append(cands[0] if cands else tok)
        return unmask(" ".join(out), kept)


# ---------------------------------------------------------------------------
# Stage 2 — translation
# ---------------------------------------------------------------------------
class Translator:
    def __init__(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit import IndicProcessor

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(C.NMT_MODEL, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(C.NMT_MODEL, trust_remote_code=True)
        self.model.to(C.device()).eval()
        self.ip = IndicProcessor(inference=True)

    def batch(self, texts: list[str]) -> list[str]:
        prepared = self.ip.preprocess_batch(
            texts, src_lang=C.NMT_SRC_TAG, tgt_lang=C.NMT_TGT_TAG)
        enc = self.tok(prepared, truncation=True, padding=True,
                       max_length=C.NMT_MAX_LEN, return_tensors="pt").to(C.device())
        with self.torch.no_grad():
            gen = self.model.generate(**enc, num_beams=C.NMT_BEAM,
                                      max_length=C.NMT_MAX_LEN, num_return_sequences=1)
        decoded = self.tok.batch_decode(gen.detach().cpu(), skip_special_tokens=True)
        return self.ip.postprocess_batch(decoded, lang=C.NMT_TGT_TAG)

    def __call__(self, text: str) -> str:
        return self.batch([text])[0]


# ---------------------------------------------------------------------------
class NormalizationPipeline:
    """Convenience wrapper: code-mixed text in, English out."""

    def __init__(self, en_vocab: Path | None = None):
        self.translit = Transliterator(en_vocab)
        self.translate = Translator()

    def __call__(self, text: str) -> str:
        return self.translate(self.translit(text))


def normalize_corpus(inp: Path, out: Path, en_vocab: Path | None = None, batch_size: int = 16):
    df = pd.read_csv(inp)
    if "comment" not in df.columns:
        raise ValueError("input CSV must contain a 'comment' column")

    translit = Transliterator(en_vocab)
    df["translit"] = [translit(str(t)) for t in tqdm(df["comment"], desc="IndicXlit")]

    translator = Translator()
    texts, translated = df["translit"].astype(str).tolist(), []
    for i in tqdm(range(0, len(texts), batch_size), desc="IndicTrans2"):
        translated.extend(translator.batch(texts[i:i + batch_size]))
    df["translated"] = translated

    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("columns: comment (code-mixed) | translit (Kannada script) | translated (English)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=C.DATA / "hastika.csv")
    ap.add_argument("--output", type=Path, default=C.DATA / "normalized.csv")
    ap.add_argument("--en_vocab", type=Path, default=None,
                    help="one English word per line; recommended for a real run")
    ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args()
    normalize_corpus(a.input, a.output, a.en_vocab, a.batch_size)
