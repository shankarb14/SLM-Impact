# Hate Speech Detection in Kannada–English Code-Mixed Text

Implementation of the proposed method: a two-stage normalization pipeline
followed by a frozen transformer encoder with a Bi-LSTM classification head.

```
code-mixed comment
   -> IndicXlit          transliteration to Kannada script      (Fig. 1)
   -> IndicTrans2        translation to English                 (Fig. 2)
   -> frozen encoder     contextual token embeddings            (Fig. 3)
   -> Bi-LSTM            sequence-level aggregation
   -> dropout / dense / sigmoid or softmax
```

Supports both classification settings — binary (hate vs. non-hate) and
fine-grained six-class — on either code-mixed or translated input, with any of
eight encoders sharing the same downstream architecture.

## Files

| File | Contents |
|---|---|
| `config.py` | Encoder registry, hyperparameters (Tables 3–4), normalization settings |
| `normalization.py` | IndicXlit + IndicTrans2 with word-level LID (Sect. III-C) |
| `model.py` | `HateSpeechClassifier`: frozen encoder + Bi-LSTM (Sect. III-E) |
| `dataset.py` | Corpus loading, stratified 70:10:20 split, batching |
| `train.py` | Training loop with early stopping |
| `predict.py` | Algorithm 1, end to end from a raw comment |

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('words')"
```

Place the corpus at `data/hastika.csv` with columns
`comment, binary_label, fine_label`. HASTIKA is available from Kavatagi &
Rachh (2025), doi:10.1007/s10579-025-09836-1.

## Use

```bash
# 1. normalize once; the result is cached and reused
python normalization.py --input data/hastika.csv --output data/normalized.csv

# 2. train
python train.py --encoder MobileBERT --task binary     --modality codemixed
python train.py --encoder DistilBERT --task multiclass --modality translated

# 3. predict
python predict.py --checkpoint checkpoints/binary_codemixed_MobileBERT.pt \
                  --text "Intaha loafergala karanadindale samaja nashavaguttide"
```

Encoders: `BERT`, `mBERT`, `HateBERT`, `XLM-R`, `ALBERT`, `TinyBERT`,
`MobileBERT`, `DistilBERT`.

Use `--no_bilstm` to replace the Bi-LSTM with mean pooling over the encoder
output.

## Notes

Only the Bi-LSTM, dense, and classification layers are trained; encoder weights
stay frozen. This is why the learning rate of 1e-3 is appropriate — it applies
to randomly initialised layers, not to fine-tuning pretrained weights.

Normalization is an offline batch stage. `predict.py` loads it lazily and only
for checkpoints trained on translated input, so a code-mixed model never pays
the IndicTrans2 cost.

Word-level language identification runs before transliteration so that genuine
English spans in code-mixed input are preserved rather than transliterated.
Supply `--en_vocab` with a domain-appropriate word list for best results.
