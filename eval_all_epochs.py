#!/usr/bin/env python
"""
Validate every epoch checkpoint of a TriDet (+ATM) run on BOTH the train and
test/val splits and plot the results.

Standalone: drop this file in the repo root (next to train.py / eval.py).
It imports the repo's own libs and modifies nothing.

Example
-------
python eval_all_epochs.py configs/thumos_i3d.yaml ckpt/thumos_i3d_run1
python eval_all_epochs.py configs/thumos_i3d.yaml ckpt/thumos_i3d_run1 \
       --out_dir eval_curves --every 2 --no_ema

Outputs (in --out_dir, default <ckpt_dir>/epoch_eval)
-----------------------------------------------------
results.json          per-epoch metrics (cached; re-runs skip finished epochs)
results.csv           same, flat table
avg_mAP.png           average mAP, train vs test
mAP_per_tiou.png      mAP at each tIoU threshold, train vs test
cls_accuracy.png      action-level top-1 classification accuracy, train vs test
"""
import argparse
import csv
import glob
import json
import os
import re
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

# ---- repo imports (same ones eval.py uses) ---------------------------------
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import ANETdetection, fix_random_seed

try:  # present in TriDet / ActionFormer
    from libs.utils import postprocess_results
except ImportError:  # pragma: no cover
    postprocess_results = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def epoch_of(path):
    m = re.search(r"epoch_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def find_checkpoints(ckpt_dir, every, first, last):
    files = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth.tar"))
    files = sorted(files, key=epoch_of)
    files = [f for f in files if epoch_of(f) >= first and (last < 0 or epoch_of(f) <= last)]
    return [f for f in files if epoch_of(f) % every == 0]


def build_loader(cfg, split_key):
    ds = make_dataset(cfg["dataset_name"], False, cfg[split_key], **cfg["dataset"])
    loader = make_data_loader(ds, False, None, 1, cfg["loader"]["num_workers"])
    return ds, loader


def build_evaluator(ds):
    attrs = ds.get_attributes()
    return ANETdetection(ds.json_file, ds.split[0], tiou_thresholds=attrs["tiou_thresholds"])


@torch.no_grad()
def run_inference(loader, model):
    """Same collection logic as libs.utils.valid_one_epoch, minus the scoring."""
    results = {"video-id": [], "t-start": [], "t-end": [], "label": [], "score": []}
    model.eval()
    for video_list in loader:
        output = model(video_list)
        for out in output:
            n = out["segments"].shape[0]
            if n == 0:
                continue
            results["video-id"].extend([out["video_id"]] * n)
            results["t-start"].append(out["segments"][:, 0])
            results["t-end"].append(out["segments"][:, 1])
            results["label"].append(out["labels"])
            results["score"].append(out["scores"])
    for k in ("t-start", "t-end", "label", "score"):
        results[k] = torch.cat(results[k]).numpy() if results[k] else np.array([])
    return results


def classification_accuracy(evaluator, results, tiou=0.5):
    """
    Action-level top-1 accuracy: for each GT segment, take the highest-scoring
    prediction in the same video overlapping it with tIoU >= `tiou` (class
    agnostic) and check whether its label equals the GT label. GTs with no
    overlapping prediction count as wrong.
    """
    gt = evaluator.ground_truth
    vids = np.asarray(results["video-id"])
    correct, total = 0, 0
    for vid, g in gt.groupby("video-id"):
        mask = vids == vid
        p_s = results["t-start"][mask]
        p_e = results["t-end"][mask]
        p_l = results["label"][mask]
        p_sc = results["score"][mask]
        for _, row in g.iterrows():
            total += 1
            if p_s.size == 0:
                continue
            inter = np.clip(np.minimum(p_e, row["t-end"]) - np.maximum(p_s, row["t-start"]), 0, None)
            union = (p_e - p_s) + (row["t-end"] - row["t-start"]) - inter
            iou = inter / np.maximum(union, 1e-8)
            ok = iou >= tiou
            if not ok.any():
                continue
            best = np.argmax(np.where(ok, p_sc, -np.inf))
            correct += int(int(p_l[best]) == int(row["label"]))
    return correct / max(total, 1)


def evaluate_split(loader, evaluator, model, cfg, tiou_for_acc):
    results = run_inference(loader, model)
    raw = {k: (v.copy() if isinstance(v, np.ndarray) else list(v)) for k, v in results.items()}

    scored = results
    ext = cfg["test_cfg"].get("ext_score_file")
    # Only re-score with external classifier scores when a file is configured
    # (mirrors the repo's own valid_one_epoch); otherwise use raw predictions.
    if postprocess_results is not None and isinstance(ext, str) and ext:
        scored = postprocess_results(results, ext)
    import pandas as pd
    df = pd.DataFrame(scored) if not hasattr(scored, "columns") else scored

    # ANETdetection.evaluate -> (mAP per tIoU, average mAP, mRecall)
    # Some forks return (mAP, avg_mAP), stock ActionFormer/TriDet returns 3 values
    ev = evaluator.evaluate(df, verbose=False)
    mAP, avg_mAP = ev[0], ev[1]
    acc = classification_accuracy(evaluator, raw, tiou=tiou_for_acc)
    out = {
        "avg_mAP": float(avg_mAP) * 100,
        "mAP_per_tiou": [float(x) * 100 for x in np.atleast_1d(mAP)],
        "cls_acc": float(acc) * 100,
    }

    # Per-class AP, shape [n_tiou, n_classes]. ActionFormer-style evaluators keep
    # it in `.ap` after evaluate(); otherwise recompute it with the wrapper.
    ap = getattr(evaluator, "ap", None)
    if ap is None and hasattr(evaluator, "wrapper_compute_average_precision"):
        ap = evaluator.wrapper_compute_average_precision(df)
    if ap is not None:
        ap = np.asarray(ap, dtype=float) * 100
        out["ap_per_class_tiou"] = ap.tolist()
        out["ap_per_class"] = ap.mean(axis=0).tolist()  # averaged over tIoU thresholds
    return out


def class_names(evaluator, n):
    """Class names ordered by class index (falls back to class_0, class_1, ...)."""
    idx = getattr(evaluator, "activity_index", None)
    if isinstance(idx, dict) and len(idx) == n:
        names = [None] * n
        for name, i in idx.items():
            names[int(i)] = str(name)
        if all(names):
            return names
    return [f"class_{i}" for i in range(n)]


def load_weights(model, ckpt_path, use_ema):
    ckpt = torch.load(ckpt_path, map_location=lambda s, l: s.cuda(torch.cuda.current_device()))
    key = "state_dict_ema" if (use_ema and "state_dict_ema" in ckpt) else "state_dict"
    model.load_state_dict(ckpt[key])
    return key


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
SPLITS = ("train", "test", "unseen")


def present_splits(results):
    """Splits that have data in at least one epoch (keeps old caches working)."""
    return [s for s in SPLITS if any(s in r for r in results.values())]


def make_plots(results, tious, out_dir, classes=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = present_splits(results)

    def series(split, getter):
        eps = sorted(int(e) for e in results if split in results[e])
        return eps, [getter(results[str(e)][split]) for e in eps]

    # 1. average mAP
    plt.figure(figsize=(8, 5))
    for s in splits:
        eps, ys = series(s, lambda r: r["avg_mAP"])
        plt.plot(eps, ys, "-o", label=s)
    plt.xlabel("epoch"); plt.ylabel("average mAP (%)"); plt.grid(alpha=.3); plt.legend()
    plt.ylim(0, 100)
    plt.title("Average mAP per epoch")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "avg_mAP.png"), dpi=150); plt.close()

    # 2. per-tIoU mAP
    n = len(tious)
    cols = min(n, 4); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for i, t in enumerate(tious):
        ax = axes[i // cols][i % cols]
        for s in splits:
            eps, ys = series(s, lambda r: r["mAP_per_tiou"][i])
            ax.plot(eps, ys, label=s)
        ax.set_title(f"mAP @ tIoU {t:g}"); ax.grid(alpha=.3)
        ax.set_ylim(0, 100)
        if i == 0:
            ax.legend()
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.supxlabel("epoch"); fig.supylabel("mAP (%)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "mAP_per_tiou.png"), dpi=150); plt.close(fig)

    # 4. per-class mAP (averaged over tIoU thresholds), one panel per class
    if classes and any("ap_per_class" in results[e].get(s, {}) for e in results for s in splits):
        n = len(classes)
        cols = min(n, 4); rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
        for i, name in enumerate(classes):
            ax = axes[i // cols][i % cols]
            for s in splits:
                eps = sorted(int(e) for e in results if "ap_per_class" in results[e].get(s, {}))
                ys = [results[str(e)][s]["ap_per_class"][i] for e in eps]
                ax.plot(eps, ys, label=s)
            ax.set_title(name, fontsize=10); ax.set_ylim(0, 100); ax.grid(alpha=.3)
            if i == 0:
                ax.legend()
        for j in range(n, rows * cols):
            axes[j // cols][j % cols].axis("off")
        fig.supxlabel("epoch"); fig.supylabel("mAP averaged over tIoU (%)")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "mAP_per_class.png"), dpi=150); plt.close(fig)

    # 3. classification accuracy
    plt.figure(figsize=(8, 5))
    for s in splits:
        eps, ys = series(s, lambda r: r["cls_acc"])
        plt.plot(eps, ys, "-o", label=s)
    plt.xlabel("epoch"); plt.ylabel("top-1 action accuracy (%)"); plt.grid(alpha=.3); plt.legend()
    plt.ylim(0, 100)
    plt.title("Action-level classification accuracy per epoch")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "cls_accuracy.png"), dpi=150); plt.close()


def write_csv(results, tious, path):
    epochs = sorted(int(e) for e in results)
    splits = present_splits(results)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        head = ["epoch"]
        for s in splits:
            head += [f"{s}_avg_mAP", f"{s}_cls_acc"] + [f"{s}_mAP@{t:g}" for t in tious]
        w.writerow(head)
        for e in epochs:
            row = [e]
            for s in splits:
                r = results[str(e)].get(s)
                if r is None:
                    row += [""] * (2 + len(tious))
                else:
                    row += [r["avg_mAP"], r["cls_acc"]] + r["mAP_per_tiou"]
            w.writerow(row)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Validate all epochs on train + test splits")
    p.add_argument("config", type=str, help="yaml config used for training")
    p.add_argument("ckpt_dir", type=str, help="folder with epoch_XXX.pth.tar files")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--train_split_key", type=str, default="train_split", help="cfg key for the train split")
    p.add_argument("--test_split_key", type=str, default="val_split", help="cfg key for the test split")
    p.add_argument("--unseen_split_key", type=str, default="unseen_split", help="cfg key for the unseen split")
    p.add_argument("--every", type=int, default=1, help="evaluate every Nth epoch")
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--last", type=int, default=-1)
    p.add_argument("--no_ema", action="store_true", help="use raw weights instead of EMA weights")
    p.add_argument("--acc_tiou", type=float, default=0.5, help="tIoU used for classification accuracy")
    p.add_argument("--force", action="store_true", help="ignore cached results.json")
    p.add_argument("--plot_only", action="store_true", help="only redraw plots from results.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = args.out_dir or os.path.join(args.ckpt_dir, "epoch_eval")
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "results.json")

    cache = {"tious": None, "results": {}}
    if os.path.exists(res_path) and not args.force:
        with open(res_path) as f:
            cache = json.load(f)

    if not args.plot_only:
        ckpts = find_checkpoints(args.ckpt_dir, args.every, args.first, args.last)
        if not ckpts:
            raise SystemExit(f"No epoch_*.pth.tar found in {args.ckpt_dir}")
        print(f"Found {len(ckpts)} checkpoints")

        cudnn.benchmark = False
        cudnn.deterministic = True
        fix_random_seed(0, include_cuda=True)

        split_keys = {
            "train": args.train_split_key,
            "test": args.test_split_key,
            "unseen": args.unseen_split_key,
        }
        loaders = {}  # name -> (loader, evaluator)
        for name, key in split_keys.items():
            if key not in cfg:
                print(f"WARNING: '{key}' not in config, skipping the {name} split")
                continue
            ds, loader = build_loader(cfg, key)
            loaders[name] = (loader, build_evaluator(ds))
            if name == "test" or "tious" not in cache or not cache["tious"]:
                cache["tious"] = [float(t) for t in ds.get_attributes()["tiou_thresholds"]]
        model = make_meta_arch(cfg["model_name"], **cfg["model"])
        model = nn.DataParallel(model, device_ids=cfg["devices"])

        for ck in ckpts:
            ep = str(epoch_of(ck))
            r = cache["results"].setdefault(ep, {})
            # evaluate splits not yet cached, or cached before per-class AP was recorded
            todo = [s for s in loaders
                    if s not in r or ("ap_per_class" not in r[s] and not cache.get("no_per_class"))]
            if not todo:
                print(f"epoch {ep}: cached, skipping")
                continue
            t0 = time.time()
            key = load_weights(model, ck, use_ema=not args.no_ema)
            r["weights"] = key
            for s in todo:
                loader, evaluator = loaders[s]
                r[s] = evaluate_split(loader, evaluator, model, cfg, args.acc_tiou)
                if "ap_per_class" in r[s]:
                    if not cache.get("classes"):
                        cache["classes"] = class_names(evaluator, len(r[s]["ap_per_class"]))
                else:
                    cache["no_per_class"] = True
                    print("WARNING: evaluator exposes no per-class AP; skipping per-class graph")
            summary = " | ".join(
                f"{s} mAP {r[s]['avg_mAP']:.2f} acc {r[s]['cls_acc']:.2f}" for s in SPLITS if s in r)
            print(f"epoch {ep:>3} [{key}] | {summary} | {time.time() - t0:.0f}s")
            with open(res_path, "w") as f:  # save after every epoch so a crash loses nothing
                json.dump(cache, f, indent=2)

    if not cache["results"]:
        raise SystemExit("No results to plot.")
    write_csv(cache["results"], cache["tious"], os.path.join(out_dir, "results.csv"))
    make_plots(cache["results"], cache["tious"], out_dir, cache.get("classes"))
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
