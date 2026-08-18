

import argparse
import time

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score

import config as C
from dataset import build_loaders
from model import HateSpeechClassifier, count_parameters, get_tokenizer


def run_epoch(model, loader, loss_fn, optimizer=None):
    
    training = optimizer is not None
    model.train(training)
    total, seen = 0.0, 0

    for batch in loader:
        ids = batch["input_ids"].to(C.device())
        mask = batch["attention_mask"].to(C.device())
        y = batch["label"].to(C.device())
        y = y.float() if model.n_classes == 2 else y.long()

        with torch.set_grad_enabled(training):
            loss = loss_fn(model(ids, mask), y)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total += loss.item() * y.size(0)
        seen += y.size(0)

    return total / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, labels):
    model.eval()
    preds, golds = [], []
    for batch in loader:
        p, _ = model.predict(batch["input_ids"].to(C.device()),
                             batch["attention_mask"].to(C.device()))
        preds += p.cpu().tolist()
        golds += batch["label"].tolist()
    macro = f1_score(golds, preds, average="macro", zero_division=0)
    report = classification_report(golds, preds, target_names=labels,
                                   digits=4, zero_division=0)
    return macro, report, preds, golds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(C.ENCODERS))
    ap.add_argument("--task", required=True, choices=["binary", "multiclass"])
    ap.add_argument("--modality", required=True, choices=["codemixed", "translated"])
    ap.add_argument("--corpus", default=None, help="defaults to data/normalized.csv")
    ap.add_argument("--no_bilstm", action="store_true", help="mean-pool instead (ablation)")
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()

    C.set_seed(a.seed)
    n_classes = len(C.LABELS[a.task])

    print(f"\n{a.encoder} | {a.task} | {a.modality} | seed {a.seed}")
    tokenizer = get_tokenizer(a.encoder)
    loaders, labels = build_loaders(a.task, a.modality, tokenizer,
                                    corpus_path=a.corpus, seed=a.seed)

    model = HateSpeechClassifier(a.encoder, n_classes,
                                 use_bilstm=not a.no_bilstm).to(C.device())
    total, trainable = count_parameters(model)
    print(f"  parameters: {total:,} total, {trainable:,} trainable "
          f"({100 * trainable / total:.1f}%)\n")

    loss_fn = nn.BCEWithLogitsLoss() if n_classes == 2 else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=C.HP["lr"], betas=C.HP["adam_betas"])

    best_loss, best_state, patience = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(1, C.HP["max_epochs"] + 1):
        tr = run_epoch(model, loaders["train"], loss_fn, optimizer)
        va = run_epoch(model, loaders["val"], loss_fn)
        print(f"  epoch {epoch:2d}  train {tr:.4f}  val {va:.4f}")

        if va < best_loss - 1e-5:
            best_loss, patience = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= C.HP["early_stopping_patience"]:
                print(f"  early stopping at epoch {epoch}")
                break

    minutes = (time.time() - t0) / 60
    if best_state:
        model.load_state_dict(best_state)

    macro, report, _, _ = evaluate(model, loaders["test"], labels)
    print(f"\n  training time: {minutes:.1f} min")
    print(f"  test Macro-F1: {macro:.4f}\n")
    print(report)

    tag = f"{a.task}_{a.modality}_{a.encoder}" + ("" if not a.no_bilstm else "_nobilstm")
    path = C.CKPT / f"{tag}.pt"
    torch.save({"state_dict": model.state_dict(), "encoder": a.encoder,
                "task": a.task, "modality": a.modality, "labels": labels,
                "use_bilstm": not a.no_bilstm, "hp": C.HP}, path)
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
