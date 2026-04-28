import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Αποτελέσματα από το benchmark
tasks = [
    "Multiply\n435×928",
    "Multiply\n5000×3900", 
    "Multiply\n123×456",
    "Code\n2**10",
    "Code\nsum(1..100)",
    "Code\nlen(str)",
    "Reasoning\nCapital",
    "Reasoning\nLeap year"
]

times = [0.65, 4.82, 19.11, 9.66, 11.81, 10.75, 50.47, 41.57]
success = [True, False, True, True, True, True, False, False]

colors = ["#1D9E75" if s else "#D85A30" for s in success]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#F8F8F6")

# ─── CHART 1: Response Time ───
bars = ax1.bar(tasks, times, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
ax1.set_title("Response Time per Task", fontsize=14, fontweight="bold", pad=15)
ax1.set_ylabel("Time (seconds)", fontsize=11)
ax1.set_ylim(0, 60)
ax1.set_facecolor("#F8F8F6")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.tick_params(axis="x", labelsize=8)

# Τιμές πάνω από κάθε bar
for bar, t in zip(bars, times):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f"{t}s", ha="center", va="bottom", fontsize=8, fontweight="bold")

# Legend
pass_patch = mpatches.Patch(color="#1D9E75", label="PASS")
fail_patch = mpatches.Patch(color="#D85A30", label="FAIL")
ax1.legend(handles=[pass_patch, fail_patch], loc="upper left", fontsize=10)

# ─── CHART 2: Success Rate per Category ───
categories = ["Multiplication", "Code Execution", "Reasoning", "Overall"]
passed =     [2,               3,                0,            5]
total =      [3,               3,                2,            8]
rates = [p/t*100 for p, t in zip(passed, total)]

bar_colors = ["#378ADD", "#378ADD", "#378ADD", "#534AB7"]
bars2 = ax2.bar(categories, rates, color=bar_colors, edgecolor="white", linewidth=0.8, width=0.5)
ax2.set_title("Success Rate by Category", fontsize=14, fontweight="bold", pad=15)
ax2.set_ylabel("Success Rate (%)", fontsize=11)
ax2.set_ylim(0, 120)
ax2.set_facecolor("#F8F8F6")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.axhline(y=62, color="#534AB7", linestyle="--", linewidth=1, alpha=0.5)

# Labels
for bar, rate, p, t in zip(bars2, rates, passed, total):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f"{int(rate)}%\n({p}/{t})", ha="center", va="bottom", 
             fontsize=10, fontweight="bold")

plt.suptitle("Autonomous AI Agent — Benchmark Results", 
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("benchmark_results.png", dpi=150, bbox_inches="tight",
            facecolor="#F8F8F6")
plt.show()
print("✅ Αποθηκεύτηκε ως benchmark_results.png")