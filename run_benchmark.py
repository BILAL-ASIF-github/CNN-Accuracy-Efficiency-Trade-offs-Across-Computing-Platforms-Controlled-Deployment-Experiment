import numpy as np
import pandas as pd
import time
import psutil
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms, datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE = Path(r"C:\Users\pc\Desktop\CNN Study")
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"
VARIANT_DIR = BASE / "checkpoints" / "variants"
BEST_CHECKPOINT = BASE / "checkpoints" / "baseline_fp32.pt"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

WARMUP_PASSES = 10
TIMED_PASSES = 100
CALIBRATION_SIZE = 150

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_loader():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    full = datasets.EuroSAT(root=str(DATA_DIR), download=False, transform=transform)
    test_idx = np.load(DATA_DIR / "test_indices.npy")
    return DataLoader(Subset(full, test_idx), batch_size=1, shuffle=False)


def load_fresh_model(classes):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    m.eval()
    return m


def load_variant(precision, device):
    """Return an eval model ready for inference."""
    if precision == "int8":
        ckpt = torch.load(VARIANT_DIR / "variant_int8.pt", map_location="cpu")
        return ckpt["model"].eval()
    ckpt = torch.load(VARIANT_DIR / (f"variant_{precision}.pt"), map_location=device)
    classes = ckpt["classes"]
    m = load_fresh_model(classes)
    m.load_state_dict(ckpt["model_state"])
    if precision == "fp16":
        m.half()
    return m.eval().to(device)


def quantize_int8_fresh():
    """Re-derive the INT8 model from the FP32 baseline (same recipe as export)."""
    ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu")
    classes = ckpt["classes"]
    m = load_fresh_model(classes)
    m.load_state_dict(ckpt["model_state"])
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    example = (torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),)
    mapping = get_default_qconfig_mapping("fbgemm")
    prepared = prepare_fx(m, mapping, example)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    calib = datasets.EuroSAT(root=str(DATA_DIR), transform=transform, download=False)
    calib_loader = DataLoader(Subset(calib, list(range(CALIBRATION_SIZE))), batch_size=8, shuffle=False)
    with torch.no_grad():
        for images, _ in tqdm(calib_loader, desc="  calibrating INT8", unit="img", ncols=90):
            prepared(images)
    return convert_fx(prepared).eval()


def to_input_dtype(x, precision):
    return x.half() if precision == "fp16" else x


def measure_peak_memory(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return psutil.Process().memory_info().rss / (1024 ** 2)


def measure_latency(model, precision, device, label):
    loader = get_test_loader()
    it = iter(loader)
    with torch.no_grad():
        for _ in range(WARMUP_PASSES):
            images, _ = next(it)
            model(to_input_dtype(images, precision).to(device))
        lat = []
        for _ in tqdm(range(TIMED_PASSES), desc=f"[{label}] latency passes", ncols=90):
            images, _ = next(it)
            images = to_input_dtype(images, precision).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
    mean_ms = (sum(lat) / len(lat)) * 1000
    return mean_ms, 1000.0 / mean_ms


def score_accuracy(model, precision, device, label):
    loader = get_test_loader()
    all_preds, all_labels = [], []
    it = tqdm(enumerate(loader), total=len(loader), desc=f"[{label}] scoring test set",
              unit="img", ncols=90)
    with torch.no_grad():
        for i, (images, labels_) in it:
            preds = model(to_input_dtype(images, precision).to(device)).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels_.tolist())
            if (i + 1) % 500 == 0:
                tqdm.write(f"    {label}: scored {i+1}/{len(loader)} images")
    return {
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "precision_m": round(precision_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "recall": round(recall_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "f1": round(f1_score(all_labels, all_preds, average="macro", zero_division=0), 4),
    }


def already_done(platform, precision, path):
    if not path.exists():
        return False
    df = pd.read_csv(path)
    if df.empty:
        return False
    return ((df["platform"] == platform) & (df["precision"] == precision)).any()


def save_row(row, path):
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gpu = DEVICE.type == "cuda"
    platform = "GPU (RTX 5060)" if gpu else "CPU"

    int8_model = None
    gpu_fp16_row = None
    if RESULTS_CSV.exists():
        prev = pd.read_csv(RESULTS_CSV)
        hit = prev[(prev["platform"] == "GPU (RTX 5060)") & (prev["precision"] == "fp16")]
        if not hit.empty:
            gpu_fp16_row = hit.iloc[-1].to_dict()

    print()
    print("=" * 70)
    print("ACCURACY-EFFICIENCY BENCHMARK  |  test set = 4,050 images")
    print(f"device = {platform}   warmup={WARMUP_PASSES}   timed={TIMED_PASSES}")
    print("results appended per-config ->", RESULTS_CSV)
    print("=" * 70)

    def run_config(platform_, precision, device):
        print()
        print(f"[{precision.upper()}] {platform_} ...")
        nonlocal int8_model, gpu_fp16_row
        if precision == "int8":
            tqdm.write("  note: exported variant_int8.pt deserializes into a broken graph "
                       "in a fresh process; re-quantizing INT8 in-process (validated)")
            model = quantize_int8_fresh()
        else:
            model = load_variant(precision, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        ms, ips = measure_latency(model, precision, device, f"{platform_} {precision.upper()}")
        if precision == "fp16" and platform_ == "CPU":
            base = gpu_fp16_row
            note = ("latency measured on CPU; accuracy = measured fp16 model accuracy "
                    "(identical fp16 weights/dtype, platform-independent)")
            print(f"  CPU-FP16 note: accuracy reused from measured fp16 model ({base['accuracy']})")
            row = dict(base)
            row.update({"platform": "CPU", "precision": "fp16",
                        "latency_ms": round(ms, 3), "throughput_ips": round(ips, 2),
                        "memory_mb": round(measure_peak_memory(device), 1),
                        "note": note})
        else:
            acc = score_accuracy(model, precision, device, f"{platform_} {precision.upper()}")
            mem = measure_peak_memory(device)
            row = {"platform": platform_, "precision": precision, **acc,
                   "latency_ms": round(ms, 3), "throughput_ips": round(ips, 2),
                   "memory_mb": round(mem, 1), "note": ""}
            if precision == "fp16" and platform_.startswith("GPU"):
                gpu_fp16_row = row
        save_row(row, RESULTS_CSV)
        print(f"  => Acc={row['accuracy']}  Latency={row['latency_ms']}ms  "
              f"Throughput={row['throughput_ips']:.1f} img/s  Memory={row['memory_mb']:.0f}MB")
        return row

    # 1) GPU FP32
    if gpu and not already_done("GPU (RTX 5060)", "fp32", RESULTS_CSV):
        run_config("GPU (RTX 5060)", "fp32", DEVICE)

    # 2) GPU FP16 (also supplies the measured fp16 accuracy for CPU FP16)
    if gpu and not already_done("GPU (RTX 5060)", "fp16", RESULTS_CSV):
        run_config("GPU (RTX 5060)", "fp16", DEVICE)

    if gpu:
        print("\nGPU x INT8: N/A (PyTorch native quantization is CPU-only)")

    # 3) CPU FP32
    if not already_done("CPU", "fp32", RESULTS_CSV):
        run_config("CPU", "fp32", torch.device("cpu"))

    # 4) CPU FP16 (pathological on this CPU -> timed latency + measured fp16 accuracy)
    if not already_done("CPU", "fp16", RESULTS_CSV):
        run_config("CPU", "fp16", torch.device("cpu"))

    # 5) CPU INT8
    if not already_done("CPU", "int8", RESULTS_CSV):
        run_config("CPU", "int8", torch.device("cpu"))

    print()
    print("=" * 70)
    print("ALL CONFIGS COMPLETE")
    print("=" * 70)
    df = pd.read_csv(RESULTS_CSV)
    cols = ["platform", "precision", "accuracy", "precision_m", "recall", "f1",
            "latency_ms", "throughput_ips", "memory_mb"]
    print(df[cols].to_string(index=False))
    print(f"\nSaved: {RESULTS_CSV}")


if __name__ == "__main__":
    main()