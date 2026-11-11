"""Render README figures: the two-stage architecture and what the preprocessing step feeds the models."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

import sys  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.inference import Labels, preprocess  # noqa: E402

OUT = ROOT / "docs" / "images"
INK, MUTED, ACCENT, GREEN, AMBER = "#0e1a2b", "#5b6676", "#2747d6", "#1f8a4c", "#c97d06"


def architecture(labels: Labels) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", lw=1.6,
                                    edgecolor=color, facecolor="white"))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=10.5, weight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=7.8, color=MUTED)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.3, color=MUTED))

    box(0.2, 1.8, 1.9, 1.0, "POST /predict", "leaf photo, any size")
    box(2.6, 1.8, 2.0, 1.0, "Preprocess", "EXIF fix · RGB · 256×256")
    box(5.1, 1.8, 2.0, 1.0, "Router CNN", "which crop?", color=ACCENT)
    ys = {"Corn": 3.3, "Wheat": 1.8, "Rice": 0.3}
    for crop, y in ys.items():
        n = len(labels.diseases[crop])
        sub = f"{n} classes" if crop != "Wheat" else f"17 outputs · {n} labelled"
        box(7.7, y, 2.0, 1.0, f"{crop} CNN", sub, color=GREEN if crop != "Wheat" else AMBER)
        arrow((7.1, 2.3), (7.7, y + 0.5))
        arrow((9.7, y + 0.5), (10.15, 2.3))
    box(10.15, 1.8, 1.65, 1.0, "JSON", "crop · disease ·\nconfidence · top-k", color=AMBER)
    arrow((2.1, 2.3), (2.6, 2.3))
    arrow((4.6, 2.3), (5.1, 2.3))
    ax.text(6.1, 1.45, "only the selected\ncrop model runs", ha="center", fontsize=7.5, color=MUTED)
    ax.set_title("Two-stage classification: route by crop, then diagnose", loc="left", fontsize=13,
                 weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "architecture.png")
    plt.close(fig)


def preprocessing(labels: Labels) -> None:
    samples = sorted((ROOT / "assets").glob("*.jpg"))
    fig, axes = plt.subplots(2, len(samples), figsize=(2.6 * len(samples), 5.4), dpi=150)
    for i, p in enumerate(samples):
        raw = Image.open(p)
        axes[0, i].imshow(raw.convert("RGB"))
        axes[0, i].set_title(f"{p.name}\n{raw.width}×{raw.height}", fontsize=8, color=MUTED)
        batch = preprocess(p.read_bytes(), labels.image_size)
        axes[1, i].imshow(batch[0].astype("uint8"))
        axes[1, i].set_title(f"model input {batch.shape[1]}×{batch.shape[2]}", fontsize=8, color=ACCENT)
        for ax in axes[:, i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#e3e7ee")
    fig.suptitle("Sample photos before and after preprocessing", fontsize=13, weight="bold", color=INK, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "preprocessing.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    lab = Labels.load()
    architecture(lab)
    preprocessing(lab)
    print("figures written to", OUT)
