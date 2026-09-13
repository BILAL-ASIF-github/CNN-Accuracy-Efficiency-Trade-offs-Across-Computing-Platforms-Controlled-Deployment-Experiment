# CNN-Accuracy-Efficiency-Trade-offs-Across-Computing-Platforms-Controlled-Deployment-Experiment
# CNN Accuracy–Efficiency Trade-offs Across Computing Platforms

**A Literature-Motivated Controlled Deployment Experiment**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)

A single ResNet-18 is trained once on EuroSAT, frozen, and deployed **unmodified** across CPU and GPU hardware at three numeric precisions (FP32, FP16, INT8) — isolating the effect of *where* and *how precisely* a model runs from the effect of the model itself. This repository contains the full pipeline: training, precision export, benchmarking, analysis, and the resulting figures.

📄 Full paper: *CNN Accuracy–Efficiency Trade-offs Across Computing Platforms: A Literature-Motivated Controlled Deployment Experiment*, M. B. Asif, 2026. (link available once published/hosted)

---

## Key Findings

| Metric | Result |
|---|---|
| Baseline accuracy (ResNet-18, EuroSAT, FP32) | **97.78%** test accuracy |
| Accuracy spread across all 5 configurations | **0.03 percentage points** (97.70–97.73%) |
| Fastest configuration | **GPU/FP32 — 3.909 ms** |
| Slowest configuration | **CPU/FP16 — 2,473.969 ms** (≈51× slower than CPU/FP32, with no accuracy benefit) |
| GPU vs. CPU speedup (FP32) | **12.4×** |
| CPU-side INT8 vs. CPU-side FP32 | **8.9× faster, no accuracy cost** |
| Pareto-optimal configurations | **GPU/FP32** and **CPU/INT8** (2 of 5) |

Full results, per-class metrics, the Pareto analysis, and the proposed CNN Deployment Selection Framework are in the paper.

---

## Repository Structure

```
CNN-Accuracy-Efficiency-Trade-offs-Across-Computing-Platforms-Controlled-Deployment-Experiment/
│
├── notebooks/
│   ├── CNN_Accuracy_Efficiency_Study.ipynb   # Main pipeline: train ResNet-18 on GPU,
│   │                                          # export FP32/FP16/INT8 variants, benchmark
│   │                                          # CPU (desktop) + GPU, analyze results
│   └── 6_cpu_laptop_benchmark.ipynb           # Independent replication: FP32 benchmark
│                                               # on a second machine (consumer laptop)
│
├── scripts/
│   ├── export_int8_ts.py                      # Standalone INT8 export (PyTorch FX
│   │                                           # graph-mode post-training static
│   │                                           # quantization, fbgemm backend)
│   ├── run_benchmark.py                       # Reusable benchmark harness — same
│   │                                           # methodology for every platform/precision
│   └── analyze_final.py                       # Builds the results table, computes the
│                                               # Pareto frontier, generates figures
│
├── results/
│   └── fp32_benchmark_result.csv              # Laptop-CPU FP32 replication output
│
├── figs/                                      # Generated figures (accuracy/latency bars,
│                                               # Pareto frontier, pipeline diagrams, etc.)
│
├── README.md
└── LICENSE
```

## Pipeline

The experiment runs as five stages, each isolating one variable while holding everything else — architecture, weights, dataset, preprocessing — fixed:

1. **Train** a ResNet-18 (ImageNet-1K pretrained) on EuroSAT once. Weights are frozen from this point on.
2. **Export** three precision variants from the identical checkpoint: FP32 (unmodified), FP16 (half precision), INT8 (PyTorch FX static quantization, `fbgemm` backend, calibrated on 150 training images).
3. **Deploy** each variant, unmodified, on every platform it supports: CPU at all three precisions, GPU at FP32/FP16 (GPU-INT8 isn't supported by PyTorch's native quantization backend).
4. **Benchmark** each configuration with the same protocol: 10 warm-up passes discarded, 100 timed passes averaged, batch size 1. Accuracy, latency, throughput, and memory are recorded for every run.
5. **Analyze**: build the results table, compute the Pareto frontier, generate the figures.

### Reproducing the results

```bash
# 1. Set up environment
pip install torch torchvision scikit-learn psutil pandas matplotlib

# 2. Run the main notebook (training + GPU/desktop-CPU benchmarking)
jupyter notebook notebooks/CNN_Accuracy_Efficiency_Study.ipynb

# 3. Export INT8 separately if needed on a new machine
python scripts/export_int8_ts.py --checkpoint <path_to_checkpoint>

# 4. Benchmark any additional platform (e.g., a second CPU machine)
python scripts/run_benchmark.py --variant <path_to_variant> --precision fp32 --device cpu

# 5. Regenerate the results table, Pareto frontier, and figures
python scripts/analyze_final.py --results-csv <path_to_combined_results.csv>
```

Check each script's `--help` output for the exact arguments it expects — flags may differ slightly between machines depending on local paths.

### Replicating on your own hardware

`6_cpu_laptop_benchmark.ipynb` is a self-contained notebook for benchmarking a second, independent machine. It expects the exported `variant_*.pt` checkpoint files (copied from wherever training happened) and auto-detects the correct quantized backend (`fbgemm` for x86, `qnnpack` for ARM) so INT8 works whether you're on a laptop, desktop, or an ARM-based cloud instance.

---

## Dataset

[EuroSAT](https://github.com/phelber/EuroSAT) (RGB), 27,000 images across 10 land-cover classes. Split 70/15/15 (train/val/test) with a fixed seed (42) for reproducibility. Images resized to 224×224, normalized with standard ImageNet statistics; training additionally uses RandomResizedCrop and random horizontal flip.

## Hardware Used

| Role | Spec |
|---|---|
| GPU / training host | NVIDIA GeForce RTX 5060 (8GB VRAM), CUDA 12.8, PyTorch 2.11.0+cu128 |
| CPU (desktop, same machine as GPU) |
| CPU (laptop, independent replication) | 11th Gen Intel Core i7-1165G7 @ 2.80GHz, 4 physical / 8 logical cores, 15.7GB RAM |

## Citation

If you use this code, dataset split, or results, please cite:

```bibtex
@misc{asif2026cnntradeoffs,
  author       = {Asif, Muhammad Bilal},
  title        = {{CNN Accuracy--Efficiency Trade-offs Across Computing Platforms:
                   A Literature-Motivated Controlled Deployment Experiment}},
  year         = {2026},
  howpublished = {\url{https://github.com/BILAL-ASIF-github/CNN-Accuracy-Efficiency-Trade-offs-Across-Computing-Platforms-Controlled-Deployment-Experiment}},
  note         = {GitHub repository}
}
```

> M. B. Asif, "CNN Accuracy–Efficiency Trade-offs Across Computing Platforms: A Literature-Motivated Controlled Deployment Experiment," GitHub repository, 2026. [Online]. Available: https://github.com/BILAL-ASIF-github/CNN-Accuracy-Efficiency-Trade-offs-Across-Computing-Platforms-Controlled-Deployment-Experiment

## License

*(TBD)*

## Author

**Muhammad Bilal Asif**
Dept. of Computer Science, NASTP Institute of Information Technology / Air University
