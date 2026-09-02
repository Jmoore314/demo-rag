"""
Optional exploratory step: project the 1024-dim embeddings down to 2D so we
can look at them, and see PCA vs t-SNE side by side to make a concrete point
about *why* you shouldn't over-trust either projection.

This is a debugging/intuition tool, not part of the pipeline -- retrieval
never looks at this plot, it works directly on the full 1024-dim vectors.
"""

import json

import matplotlib
matplotlib.use("Agg")  # no display in this environment; just write a file
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load(path: str):
    with open(path) as f:
        records = [json.loads(line) for line in f]
    embeddings = np.array([r["embedding"] for r in records])
    pages = np.array([r["page_start"] for r in records])
    return records, embeddings, pages


def main():
    records, embeddings, pages = load("data/chunks_embedded.jsonl")
    print(f"Loaded {len(records)} embeddings, shape {embeddings.shape}")

    # PCA: a linear projection onto the 2 directions of greatest variance.
    # It's "honest" -- distances in the resulting 2D plot are a real (if
    # incomplete) reflection of distances in the original 1024-dim space.
    pca_coords = PCA(n_components=2, random_state=0).fit_transform(embeddings)

    # t-SNE: a nonlinear projection that optimizes for preserving *local*
    # neighborhoods, at the cost of global distances. It tends to produce
    # visually tighter, more dramatic-looking clusters -- which is exactly
    # why it's easy to over-trust. Perplexity roughly controls neighborhood
    # size; kept modest given the chunk count here.
    tsne_coords = TSNE(n_components=2, perplexity=20, random_state=0, init="pca").fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, coords, title in [
        (axes[0], pca_coords, "PCA (linear, preserves global structure)"),
        (axes[1], tsne_coords, "t-SNE (nonlinear, local structure only)"),
    ]:
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=pages, cmap="viridis", s=22, alpha=0.85, linewidths=0,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("component 1 (unitless)")
        ax.set_ylabel("component 2 (unitless)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(scatter, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("source page number (1-140)")

    fig.suptitle(
        f"{len(records)} chunk embeddings, colored by page position in the source PDF",
        fontsize=12,
    )

    out_path = "data/embeddings_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
