"""Aggregate formal runs, compute timing/geometry diagnostics and report figures.

Everything in results/metrics/summary.json is recomputed from the per-run JSON
files and the real PNG outputs; no number is typed by hand.
"""

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
FIGURES = RESULTS / "figures"
IMAGES = RESULTS / "images"
RAW = RESULTS / "raw"


def load_runs(tags):
    runs = {}
    for tag in tags:
        path = METRICS / f"run_{tag}.json"
        if not path.exists():
            raise SystemExit("missing run json: %s" % path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "ok":
            raise SystemExit("run %s is not ok: %s" % (tag, data.get("status")))
        for key in ("sample_s", "vae_s", "total_s"):
            if data.get(key) is None:
                raise SystemExit("run %s missing %s" % (tag, key))
        runs[tag] = data
    return runs


def read_rgb(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def edge_stats(reference_path, generated_path):
    ref = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    gen = cv2.imread(str(generated_path), cv2.IMREAD_GRAYSCALE)
    if ref is None or gen is None:
        raise FileNotFoundError("cannot read %s / %s" % (reference_path, generated_path))
    if gen.shape != ref.shape:
        gen = cv2.resize(gen, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    ref_edges = cv2.Canny(ref, 40, 120)
    gen_edges = cv2.Canny(gen, 60, 160)
    if ref_edges.sum() == 0 or gen_edges.sum() == 0:
        return {"edge_mean_px": None, "edge_p90_px": None, "recall_2px": None}
    dist = cv2.distanceTransform(255 - gen_edges, cv2.DIST_L2, 5)
    values = dist[ref_edges > 0]
    return {
        "edge_mean_px": round(float(values.mean()), 3),
        "edge_p90_px": round(float(np.percentile(values, 90)), 3),
        "recall_2px": round(float((values <= 2.0).mean()), 4),
    }


def make_label(img, text, height=32):
    w, h = img.size
    canvas = Image.new("RGB", (w, h + height), (255, 255, 255))
    canvas.paste(img, (0, height))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 7), text, fill=(20, 20, 20), font=font)
    return canvas


def make_control_figure(tag):
    raw = RAW / tag
    panels = [
        (raw / f"{tag}_clay.png", "clay"),
        (raw / f"{tag}_depth.png", "depth (near=bright)"),
        (raw / f"{tag}_normal.png", "normal"),
    ]
    imgs = []
    for path, label in panels:
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as im:
            imgs.append(make_label(im.convert("RGB").resize((420, 420)), label))
    out = Image.new("RGB", (sum(i.width for i in imgs), max(i.height for i in imgs)), "white")
    x = 0
    for im in imgs:
        out.paste(im, (x, 0))
        x += im.width
    path = FIGURES / "fig_control_A.png"
    out.save(path)
    return path


def make_compare_figure(tags):
    cells = []
    for tag in tags:
        path = IMAGES / f"{tag}_photoreal.png"
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as im:
            cells.append(make_label(im.convert("RGB").resize((360, 360)), f"{tag} photoreal"))
    cols = 3
    rows = (len(cells) + cols - 1) // cols
    w = 360 * cols
    h = cells[0].height * rows
    out = Image.new("RGB", (w, h), "white")
    for i, im in enumerate(cells):
        out.paste(im, ((i % cols) * im.width, (i // cols) * im.height))
    path = FIGURES / "fig_compare.png"
    out.save(path)
    return path


def make_overlay(tag):
    clay = cv2.imread(str(RAW / tag / f"{tag}_clay.png"))
    photo = cv2.imread(str(IMAGES / f"{tag}_photoreal.png"))
    if photo.shape[:2] != clay.shape[:2]:
        photo = cv2.resize(photo, (clay.shape[1], clay.shape[0]))
    depth = cv2.imread(str(RAW / tag / f"{tag}_depth.png"), cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(depth, 40, 120)
    overlay = photo.copy()
    overlay[edges > 0] = (0, 0, 255)
    blend = cv2.addWeighted(photo, 0.65, clay, 0.35, 0)
    top = np.concatenate([clay, photo], axis=1)
    bottom = np.concatenate([overlay, blend], axis=1)
    out = np.concatenate([top, bottom], axis=0)
    out = cv2.resize(out, (out.shape[1] // 2, out.shape[0] // 2))
    path = FIGURES / f"fig_geometry_{tag}.png"
    cv2.imwrite(str(path), out)
    return path


def make_timing_figure(runs):
    tags = list(runs)
    sample = np.array([runs[t]["sample_s"] for t in tags])
    vae = np.array([runs[t]["vae_s"] for t in tags])
    total = np.array([runs[t]["total_s"] for t in tags])
    other = np.maximum(total - sample - vae, 0)
    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(8.4, 3.6), dpi=170)
    ax.bar(x, sample, label="sampling", color="#3b6ea5")
    ax.bar(x, vae, bottom=sample, label="VAE decode", color="#d08b3c")
    ax.bar(x, other, bottom=sample + vae, label="Blender/load/other", color="#9aa3ad")
    for i, t in enumerate(tags):
        ax.text(i, total[i] + max(total) * 0.015, "%.0f s" % total[i], ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_ylabel("wall time (s)")
    ax.set_title("CPU pipeline wall time by run")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURES / "fig_timing.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def make_scaling_figure(runs):
    tags = list(runs)
    work = np.array(
        [runs[t]["config"]["res"] ** 2 * runs[t]["config"]["steps"] / 1e6 for t in tags]
    )
    sample = np.array([runs[t]["sample_s"] for t in tags])
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=170)
    ax.scatter(work, sample, c="#3b6ea5")
    for t, x, y in zip(tags, work, sample):
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    if len(work) >= 2:
        k = float(np.sum(work * sample) / np.sum(work * work))
        xs = np.linspace(0, work.max() * 1.05, 50)
        ax.plot(xs, k * xs, "--", color="#d08b3c", label="least-squares k=%.3f s/Mpx-step" % k)
        ax.legend(fontsize=7)
    ax.set_xlabel("resolution^2 x steps (megapixel-steps)")
    ax.set_ylabel("sampling time (s)")
    ax.set_title("Sampling cost scaling")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = FIGURES / "fig_scaling.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", default="A,B,C,D,E")
    args = parser.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = load_runs(tags)

    geometry = {}
    for tag in tags:
        geometry[tag] = {
            "depth_edges_vs_photoreal": edge_stats(
                RAW / tag / f"{tag}_depth.png", IMAGES / f"{tag}_photoreal.png"
            ),
            "clay_edges_vs_photoreal": edge_stats(
                RAW / tag / f"{tag}_clay.png", IMAGES / f"{tag}_photoreal.png"
            ),
        }

    def delta(a, b):
        if a not in runs or b not in runs:
            return None
        return {
            "total_ratio": round(runs[a]["total_s"] / runs[b]["total_s"], 4),
            "sample_ratio": round(runs[a]["sample_s"] / runs[b]["sample_s"], 4),
            "pixel_step_ratio": round(
                (runs[a]["config"]["res"] ** 2 * runs[a]["config"]["steps"])
                / (runs[b]["config"]["res"] ** 2 * runs[b]["config"]["steps"]),
                4,
            ),
        }

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "geometry": geometry,
        "deltas": {
            "B_over_A": delta("B", "A"),
            "C_over_A": delta("C", "A"),
            "D_over_A": delta("D", "A"),
            "E_over_A": delta("E", "A"),
        },
        "figures": {},
    }

    summary["figures"]["control_A"] = str(make_control_figure("A").relative_to(ROOT))
    summary["figures"]["compare"] = str(make_compare_figure(tags).relative_to(ROOT))
    summary["figures"]["geometry_A"] = str(make_overlay("A").relative_to(ROOT))
    summary["figures"]["timing"] = str(make_timing_figure(runs).relative_to(ROOT))
    summary["figures"]["scaling"] = str(make_scaling_figure(runs).relative_to(ROOT))

    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(METRICS / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tag",
                "res",
                "steps",
                "cfg",
                "control_strength",
                "seed",
                "blender_s",
                "sample_s",
                "vae_s",
                "sd_total_s",
                "sd_wall_s",
                "total_s",
                "edge_mean_px",
                "edge_p90_px",
                "recall_2px",
            ]
        )
        for tag in tags:
            r = runs[tag]
            g = geometry[tag]["depth_edges_vs_photoreal"]
            writer.writerow(
                [
                    tag,
                    r["config"]["res"],
                    r["config"]["steps"],
                    r["config"]["cfg"],
                    r["config"]["control_strength"],
                    r["config"]["seed"],
                    r["blender_s"],
                    r["sample_s"],
                    r["vae_s"],
                    r["sd_total_s"],
                    r["sd_wall_s"],
                    r["total_s"],
                    g["edge_mean_px"],
                    g["edge_p90_px"],
                    g["recall_2px"],
                ]
            )
    print(json.dumps({"summary": str(METRICS / "summary.json"), "deltas": summary["deltas"], "geometry": geometry}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
