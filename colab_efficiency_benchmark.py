# ============================================================
# Efficiency benchmark for Table 14 (run in Google Colab, GPU runtime)
# Measures: batch-1 inference latency, batch-32 throughput,
#           peak GPU memory during one training step
# Matches the paper's setup: frozen encoder + Bi-LSTM head,
# max sequence length 128, NVIDIA T4.
#
# !pip install -q transformers torch
# ============================================================

import time, statistics, torch, torch.nn as nn
from transformers import AutoModel, AutoTokenizer

DEVICE = "cuda"
SEQ_LEN = 128
N_WARMUP, N_LAT, N_THR = 20, 100, 50
BATCH_THR = 32

MODELS = {
    "BERT-base":    "bert-base-uncased",
    "mBERT":        "bert-base-multilingual-cased",
    "HateBERT":     "GroNLP/hateBERT",
    "XLM-RoBERTa":  "xlm-roberta-base",
    "ALBERT-base":  "albert-base-v2",
    "TinyBERT4":    "huawei-noah/TinyBERT_General_4L_312D",
    "MobileBERT":   "google/mobilebert-uncased",
    "DistilBERT":   "distilbert-base-uncased",
}

# Same downstream head as the paper (Section III-D)
class EncoderBiLSTM(nn.Module):
    def __init__(self, encoder, hidden):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():   # frozen encoder, as in III-F
            p.requires_grad = False
        self.bilstm = nn.LSTM(hidden, 128, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.dense = nn.Linear(256, 64)
        self.out = nn.Linear(64, 1)           # binary head

    def forward(self, ids, mask):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        o, _ = self.bilstm(h)
        z = torch.relu(self.dense(self.drop(o[:, -1, :])))
        return self.out(z)

def dummy_batch(tok, bs):
    text = ["ee video thumba chennagide but comments nodi beku"] * bs
    enc = tok(text, padding="max_length", truncation=True,
              max_length=SEQ_LEN, return_tensors="pt")
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)

def bench(name, hf_id):
    tok = AutoTokenizer.from_pretrained(hf_id)
    enc = AutoModel.from_pretrained(hf_id)
    model = EncoderBiLSTM(enc, enc.config.hidden_size).to(DEVICE).eval()

    # ---- batch-1 latency (median of N_LAT runs, CUDA events) ----
    ids1, m1 = dummy_batch(tok, 1)
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(ids1, m1)
        torch.cuda.synchronize()
        times = []
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        for _ in range(N_LAT):
            s.record(); model(ids1, m1); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))          # milliseconds
    latency = statistics.median(times)

    # ---- batch-32 throughput ----
    ids32, m32 = dummy_batch(tok, BATCH_THR)
    with torch.no_grad():
        for _ in range(5):
            model(ids32, m32)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_THR):
            model(ids32, m32)
        torch.cuda.synchronize()
    throughput = N_THR * BATCH_THR / (time.perf_counter() - t0)

    # ---- peak GPU memory during one training step ----
    model.train()
    torch.cuda.reset_peak_memory_stats()
    ids16, m16 = dummy_batch(tok, 16)
    y = torch.rand(16, 1, device=DEVICE)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss = nn.BCEWithLogitsLoss()(model(ids16, m16), y)
    loss.backward(); opt.step(); opt.zero_grad()
    mem = torch.cuda.max_memory_allocated() / 1024**3

    del model, enc; torch.cuda.empty_cache()
    print(f"{name:14s}  latency {latency:6.1f} ms   "
          f"throughput {throughput:7.1f} /s   peak mem {mem:4.2f} GB")

print("GPU:", torch.cuda.get_device_name(0))
for n, h in MODELS.items():
    bench(n, h)
