import sys, json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"C:\Users\pc\Desktop\CNN Study")
RESULTS_DIR = BASE / "results"
FIG_DIR = BASE / "figs"
RESULTS_CSV = RESULTS_DIR / "results.csv"
FIG_DIR.mkdir(exist_ok=True)

results_df = pd.read_csv(RESULTS_CSV)
results_df["config"] = results_df["platform"] + " / " + results_df["precision"].str.upper()

print("=" * 90)
print("TABLE 4 \u2014 Accuracy, Latency, Throughput, and Memory by Configuration")
print("=" * 90)
display_cols = ["config", "accuracy", "precision_m", "recall", "f1",
                "latency_ms", "throughput_ips", "memory_mb"]
print(results_df[display_cols].to_string(index=False))
print("=" * 90)


def compute_pareto(df):
    flags = []
    for i, row in df.iterrows():
        dominated = (
            (df["latency_ms"] <= row["latency_ms"])
            & (df["accuracy"] >= row["accuracy"])
            & ((df["latency_ms"] < row["latency_ms"]) | (df["accuracy"] > row["accuracy"]))
        ).any()
        flags.append(not dominated)
    df["pareto_optimal"] = flags
    return df


results_df = compute_pareto(results_df)
frontier = results_df[results_df["pareto_optimal"]].sort_values("latency_ms")

print("Pareto-optimal configurations (for Table 5 / Section 7.3):")
print(frontier[["config", "accuracy", "latency_ms", "throughput_ips", "memory_mb"]].to_string(index=False))
print(f"\n{len(frontier)} of {len(results_df)} configurations are Pareto-optimal.")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors_acc = ["#2ecc71" if p else "#95a5a6" for p in results_df["pareto_optimal"]]
colors_lat = ["#e74c3c" if p else "#bdc3c7" for p in results_df["pareto_optimal"]]

axes[0].bar(results_df["config"], results_df["accuracy"], color=colors_acc, edgecolor="black", linewidth=0.5)
axes[0].set_title("Accuracy by Configuration")
axes[0].set_ylabel("Accuracy")
axes[0].tick_params(axis="x", rotation=55)
axes[0].set_ylim(0.8, 1.0)
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(results_df["config"], results_df["latency_ms"], color=colors_lat, edgecolor="black", linewidth=0.5)
axes[1].set_title("Latency by Configuration")
axes[1].set_ylabel("Latency (ms)")
axes[1].tick_params(axis="x", rotation=55)
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle("Figure 5 \u2013 Accuracy and Latency by Platform x Precision", fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_bars.png", dpi=150, bbox_inches="tight")
print(f"Saved: {FIG_DIR / 'fig5_bars.png'}")

fig, ax = plt.subplots(figsize=(8, 6))

dominated = results_df[~results_df["pareto_optimal"]]
optimal = results_df[results_df["pareto_optimal"]].sort_values("latency_ms")

ax.scatter(dominated["latency_ms"], dominated["accuracy"],
           s=120, color="gray", label="Dominated", edgecolors="black", linewidth=0.5, zorder=2)
ax.scatter(optimal["latency_ms"], optimal["accuracy"],
           s=200, color="red", label="Pareto-optimal", edgecolors="black", linewidth=1, zorder=3)

if len(optimal) > 1:
    ax.plot(optimal["latency_ms"], optimal["accuracy"], "r--", alpha=0.5, linewidth=1, zorder=1)

for _, row in results_df.iterrows():
    ax.annotate(row["config"], (row["latency_ms"], row["accuracy"]),
                fontsize=7, xytext=(5, 5), textcoords="offset points")

ax.set_xlabel("Latency (ms)", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Figure 6 \u2013 Accuracy-Latency Pareto Frontier", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig6_pareto.png", dpi=150, bbox_inches="tight")
print(f"Saved: {FIG_DIR / 'fig6_pareto.png'}")


def find_row(df, platform, precision):
    match = df[(df["platform"].str.contains(platform, case=False)) & (df["precision"] == precision)]
    return match.iloc[0] if len(match) > 0 else None


print("=" * 70)
print("RESULTS SUMMARY \u2014 for Sections 7, 8, 10")
print("=" * 70)

gpu_fp32 = find_row(results_df, "GPU", "fp32")
gpu_fp16 = find_row(results_df, "GPU", "fp16")
cpu_fp32 = find_row(results_df, "CPU", "fp32")
cpu_int8 = find_row(results_df, "CPU", "int8")

if gpu_fp32 is not None and gpu_fp16 is not None:
    acc_drop = gpu_fp32["accuracy"] - gpu_fp16["accuracy"]
    print(f"\nRQ1: Accuracy drop FP32->FP16 (GPU): {acc_drop:+.4f} ({acc_drop/gpu_fp32['accuracy']*100:+.2f}%)")

if cpu_fp32 is not None and cpu_int8 is not None:
    acc_drop_i8 = cpu_fp32["accuracy"] - cpu_int8["accuracy"]
    print(f"RQ1: Accuracy drop FP32->INT8 (CPU): {acc_drop_i8:+.4f} ({acc_drop_i8/cpu_fp32['accuracy']*100:+.2f}%)")

if gpu_fp32 is not None and cpu_fp32 is not None:
    speedup = cpu_fp32["latency_ms"] / gpu_fp32["latency_ms"]
    print(f"\nRQ2: GPU vs CPU speedup at FP32: {speedup:.1f}x")
    print(f"  CPU FP32:  {cpu_fp32['latency_ms']:.2f} ms")
    print(f"  GPU FP32:  {gpu_fp32['latency_ms']:.2f} ms")

if cpu_fp32 is not None and cpu_int8 is not None:
    i8_speedup = cpu_fp32["latency_ms"] / cpu_int8["latency_ms"]
    print(f"RQ2: INT8 vs FP32 on CPU: {i8_speedup:.1f}x")

print(f"\nRQ3: Pareto-optimal configurations:")
for _, row in frontier.iterrows():
    print(f"  {row['config']}: acc={row['accuracy']:.4f}, latency={row['latency_ms']:.2f}ms")

print(f"\nRQ4: Peak memory usage:")
for _, row in results_df.iterrows():
    print(f"  {row['config']:30s}: {row['memory_mb']:.0f} MB")

print("\n" + "=" * 70)
print("Use these exact numbers to fill Sections 7 (Discussion), 8 (Framework),")
print("and 10 (Conclusion). Do NOT rewrite from the intro \u2014 write from the data.")
print("=" * 70)

full_summary = {
    "results": results_df.to_dict("records"),
    "pareto_optimal": frontier[["config", "accuracy", "latency_ms"]].to_dict("records"),
}
summary_path = RESULTS_DIR / "full_summary.json"
with open(summary_path, "w") as f:
    json.dump(full_summary, f, indent=2)
print(f"\nFull summary saved to {summary_path}")
print("\nAll done. You have:")
print(f"  - Training curves: {FIG_DIR / 'training_curves.png'}")
print(f"  - Figure 5 (bars): {FIG_DIR / 'fig5_bars.png'}")
print(f"  - Figure 6 (Pareto): {FIG_DIR / 'fig6_pareto.png'}")
print(f"  - Table 4 data: {RESULTS_CSV}")
print(f"  - Full summary: {summary_path}")