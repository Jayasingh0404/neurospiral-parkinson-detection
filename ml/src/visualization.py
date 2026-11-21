"""
Visualization utilities for Exploratory Data Analysis, Training Histories,
Confusion Matrices, ROC/PR Curves, and Comparative Evaluation.
"""

import os
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image


def plot_class_distribution(stats: Dict[str, Any], output_path: str):
    """Plot overall class distribution and split distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Class distribution
    classes = list(stats["class_distribution"].keys())
    counts = [stats["class_distribution"][c] for c in classes]
    colors = ["#2563EB", "#DC2626"]
    bars1 = ax1.bar(classes, counts, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    ax1.set_title("Overall Class Distribution", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Number of Images", fontsize=12)
    ax1.set_ylim(0, max(counts) * 1.2)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{int(yval)} ({yval/sum(counts):.1%})",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Split distribution grouped by class
    splits = ["train", "valid", "test"]
    labels = ["healthy", "parkinson"]
    x = np.arange(len(splits))
    width = 0.35
    
    split_class_data = stats["split_by_class"]
    healthy_counts = [split_class_data.get("healthy", {}).get(s, 0) for s in splits]
    parkinson_counts = [split_class_data.get("parkinson", {}).get(s, 0) for s in splits]
    
    ax2.bar(x - width/2, healthy_counts, width, label="Healthy", color="#2563EB", edgecolor="black")
    ax2.bar(x + width/2, parkinson_counts, width, label="Parkinson", color="#DC2626", edgecolor="black")
    ax2.set_title("Distribution Across Dataset Splits", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.capitalize() for s in splits], fontsize=11)
    ax2.set_ylabel("Number of Images", fontsize=12)
    ax2.legend(frameon=True, facecolor="white", loc="upper right")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_image_dimensions(df, output_path: str):
    """Plot distribution of image widths, heights, and aspect ratios."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Scatter plot width vs height
    for label, color in [("healthy", "#2563EB"), ("parkinson", "#DC2626")]:
        sub_df = df[df["label"] == label]
        ax1.scatter(sub_df["width"], sub_df["height"], alpha=0.6, label=label.capitalize(), color=color, edgecolors="none", s=50)
    
    ax1.set_title("Image Dimensions (Width vs Height)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Width (px)", fontsize=11)
    ax1.set_ylabel("Height (px)", fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Aspect ratio histogram
    ax2.hist(df[df["label"] == "healthy"]["aspect_ratio"], bins=15, alpha=0.6, label="Healthy", color="#2563EB", edgecolor="black")
    ax2.hist(df[df["label"] == "parkinson"]["aspect_ratio"], bins=15, alpha=0.6, label="Parkinson", color="#DC2626", edgecolor="black")
    ax2.set_title("Aspect Ratio Distribution (Width / Height)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Aspect Ratio", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.legend(frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sample_grid(df, output_path: str, n_samples: int = 8):
    """Plot sample grid for both Healthy and Parkinson classes with dimensions."""
    fig, axes = plt.subplots(2, n_samples, figsize=(2 * n_samples, 5.5))
    
    for row_idx, label in enumerate(["healthy", "parkinson"]):
        sub_df = df[df["label"] == label].sample(n=n_samples, random_state=42)
        for col_idx, (_, row) in enumerate(sub_df.iterrows()):
            ax = axes[row_idx, col_idx]
            with Image.open(row["path"]) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"{row['subject_id']}\n{row['width']}x{row['height']}", fontsize=8)
            ax.axis("off")
            if col_idx == 0:
                ax.text(-0.25, 0.5, label.upper(), transform=ax.transAxes,
                        fontsize=12, fontweight="bold", rotation=90, va="center", ha="right",
                        color="#2563EB" if label == "healthy" else "#DC2626")
                
    plt.suptitle("Dataset Samples (Healthy vs Parkinson Spiral/Drawing Exams)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pixel_intensity_distribution(df, output_path: str, n_samples_per_class: int = 40):
    """Plot channel and grayscale pixel intensity distribution across classes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for label, color in [("healthy", "#2563EB"), ("parkinson", "#DC2626")]:
        sub_df = df[df["label"] == label].sample(n=min(n_samples_per_class, len(df[df["label"] == label])), random_state=42)
        all_pixels = []
        for _, row in sub_df.iterrows():
            with Image.open(row["path"]) as img:
                arr = np.asarray(img.convert("L")).flatten()
                all_pixels.extend(arr[::10])  # Subsample for speed
        
        ax1.hist(all_pixels, bins=64, alpha=0.5, label=label.capitalize(), color=color, density=True)
    
    ax1.set_title("Grayscale Pixel Intensity Distribution", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Pixel Value (0-255)", fontsize=11)
    ax1.set_ylabel("Normalized Density", fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    # Calculate RGB channel means over training split
    train_df = df[df["split"] == "train"]
    rgb_means = {"R": [], "G": [], "B": []}
    for _, row in train_df.iterrows():
        with Image.open(row["path"]) as img:
            arr = np.asarray(img.convert("RGB"))
            rgb_means["R"].append(np.mean(arr[:, :, 0]))
            rgb_means["G"].append(np.mean(arr[:, :, 1]))
            rgb_means["B"].append(np.mean(arr[:, :, 2]))
            
    channels = ["Red", "Green", "Blue"]
    channel_vals = [np.mean(rgb_means["R"]), np.mean(rgb_means["G"]), np.mean(rgb_means["B"])]
    channel_stds = [np.std(rgb_means["R"]), np.std(rgb_means["G"]), np.std(rgb_means["B"])]
    
    colors_rgb = ["#EF4444", "#10B981", "#3B82F6"]
    bars = ax2.bar(channels, channel_vals, yerr=channel_stds, capsize=5, color=colors_rgb, edgecolor="black", width=0.5)
    ax2.set_title("Training Set RGB Channel Mean Intensity (± 1 Std)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Mean Intensity (0-255)", fontsize=11)
    ax2.set_ylim(0, 270)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 10, f"{yval:.1f}", ha="center", va="bottom", fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_subject_leakage_map(df, output_path: str):
    """Visualizes the presence of subjects across splits showing data contamination."""
    pivot = df.groupby(["subject_id", "split"]).size().unstack(fill_value=0)
    splits = [s for s in ["train", "valid", "test"] if s in pivot.columns]
    pivot = pivot[splits]
    
    # Sort by total images and class
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values(by="total", ascending=False)
    pivot = pivot.drop(columns=["total"])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(pivot.values > 0, cmap="Blues", aspect="auto")
    
    ax.set_xticks(np.arange(len(splits)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([s.capitalize() for s in splits], fontsize=12, fontweight="bold")
    ax.set_yticklabels(pivot.index, fontsize=9)
    
    # Add count values inside cells
    for i in range(len(pivot.index)):
        for j in range(len(splits)):
            val = pivot.iloc[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", color="white" if val > 2 else "black", fontweight="bold")
                
    ax.set_title("Subject Cross-Split Contamination Audit\n(Numbers indicate image count per split)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

