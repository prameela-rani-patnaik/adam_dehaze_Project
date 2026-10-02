"""
run_all.py  -  ONE script: calculates metrics for all hybrid methods
and saves everything in table format automatically.

Run:   python run_all.py

Output folder: Hybrid_Results/Evaluation/
   metrics_tables.xlsx   (sheets: Detailed, Summary, By_Haze, Best_Method)
   Detailed.csv, Summary.csv, By_Haze.csv, Best_Method.csv
   comparison_chart.png
"""

import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# ===================== EDIT ONLY THESE 3 LINES =====================
RESULTS_DIR = Path(r"C:\Users\patna\OneDrive\Desktop\IP_1\Hybrid_Results")
GT_DIR = Path(r"C:\Users\patna\OneDrive\Desktop\IP_1\input_images\clear_image")        # clear images (optional)
HAZY_DIR = Path(r"C:\Users\patna\OneDrive\Desktop\IP_1\input_images\image1")    # hazy images (optional)
# ===================================================================

SAVE_DIR = RESULTS_DIR / "Evaluation"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
HAZE_LEVELS = ("high", "medium", "low")
HIGHER_BETTER = {"PSNR": True, "SSIM": True, "MSE": False, "CIEDE2000": False,
                 "Entropy": True, "Contrast": True, "AvgGradient": True,
                 "Colorfulness": True}


def read_rgb(p):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def haze_level(name):
    n = name.lower()
    for l in HAZE_LEVELS:
        if n.startswith(l) or f"_{l}" in n:
            return l
    return None


def find_image(folder, image_name, level=None):
    if not folder.exists():
        return None
    c = []
    for p in folder.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            s = (p.stem + " " + p.parent.name).lower()
            if image_name.lower() in s and (level is None or level in s):
                c.append(p)
    c.sort(key=lambda p: (p.stem.lower() != image_name.lower(), len(p.name)))
    return c[0] if c else None


def resize_to(img, ref):
    if img.shape[:2] != ref.shape[:2]:
        img = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    return img


def full_ref(img, gt):
    img = resize_to(img, gt)
    return {
        "PSNR": psnr_fn(gt, img, data_range=255),
        "SSIM": ssim_fn(gt, img, channel_axis=2, data_range=255),
        "MSE": float(np.mean((img.astype(float) - gt.astype(float)) ** 2)),
        "CIEDE2000": float(np.mean(deltaE_ciede2000(rgb2lab(gt), rgb2lab(img)))),
    }


def no_ref(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h = np.bincount(gray.ravel(), minlength=256) / gray.size
    h = h[h > 0]
    g = gray.astype(float)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    r, gc, b = [img[..., i].astype(float) for i in range(3)]
    rg, yb = r - gc, 0.5 * (r + gc) - b
    return {
        "Entropy": float(-np.sum(h * np.log2(h))),
        "Contrast": float(g.std()),
        "AvgGradient": float(np.mean(np.sqrt((gx ** 2 + gy ** 2) / 2))),
        "Colorfulness": float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                              + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)),
    }


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    methods = sorted(d for d in RESULTS_DIR.iterdir()
                     if d.is_dir() and d.name.lower().endswith("_results"))
    if not methods:
        raise SystemExit(f"No '*_Results' folders found in {RESULTS_DIR}")

    for mdir in methods:
        method = re.sub(r"_results$", "", mdir.name, flags=re.I)
        for idir in sorted(d for d in mdir.iterdir() if d.is_dir()):
            for f in sorted(idir.iterdir()):
                if f.suffix.lower() not in IMG_EXT or "transmission" in f.stem.lower():
                    continue
                lvl = haze_level(f.name)
                if lvl is None:
                    print(f"[skip] {f.name}: haze level not in name")
                    continue
                img = read_rgb(f)
                row = {"Method": method, "Image": idir.name, "Haze": lvl, "File": f.name}
                gt_p = find_image(GT_DIR, idir.name)
                if gt_p is not None:
                    gt = read_rgb(gt_p)
                    row.update(full_ref(img, gt))
                else:
                    print(f"[warn] no ground truth found for {idir.name}")
                row.update(no_ref(img))
                hz_p = find_image(HAZY_DIR, idir.name, lvl) if gt_p is not None else None
                if hz_p is not None:
                    base = full_ref(read_rgb(hz_p), gt)
                    row["PSNR_gain"] = row["PSNR"] - base["PSNR"]
                    row["SSIM_gain"] = row["SSIM"] - base["SSIM"]
                rows.append(row)
                print(f"[ok] {method:12s} {idir.name:10s} {lvl}")

    if not rows:
        raise SystemExit("No images evaluated. Check folder names and paths.")

    detailed = pd.DataFrame(rows).sort_values(["Method", "Image", "Haze"]).round(4)
    metrics = [m for m in HIGHER_BETTER if m in detailed.columns]
    num = detailed.select_dtypes("number").columns

    mean = detailed.groupby("Method")[num].mean()
    ranks = pd.DataFrame({m: mean[m].rank(ascending=not HIGHER_BETTER[m], method="min")
                          for m in metrics})
    summary = mean.copy()
    summary["Avg_Rank"] = ranks.mean(axis=1)
    summary = summary.sort_values("Avg_Rank").round(4)

    by_haze = detailed.groupby(["Method", "Haze"])[num].mean().round(4)

    best = pd.DataFrame({
        "Best_Method": [mean[m].idxmax() if HIGHER_BETTER[m] else mean[m].idxmin()
                        for m in metrics],
        "Best_Value": [mean[m].max() if HIGHER_BETTER[m] else mean[m].min()
                       for m in metrics],
        "Rule": ["higher is better" if HIGHER_BETTER[m] else "lower is better"
                 for m in metrics],
    }, index=metrics).round(4)

    tables = {"Detailed": detailed, "Summary": summary,
              "By_Haze": by_haze, "Best_Method": best}
    for name, t in tables.items():
        t.to_csv(SAVE_DIR / f"{name}.csv", index=name != "Detailed")
    try:
        with pd.ExcelWriter(SAVE_DIR / "metrics_tables.xlsx") as w:
            for name, t in tables.items():
                t.to_excel(w, sheet_name=name, index=name != "Detailed")
    except ModuleNotFoundError:
        print("[info] Excel file skipped. Run: pip install openpyxl  (CSV files were saved)")

    # chart
    n = len(metrics)
    cols = 4
    rws = (n + cols - 1) // cols
    fig, axes = plt.subplots(rws, cols, figsize=(4.5 * cols, 3.6 * rws))
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, metrics):
        mean[m].plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(f"{m} {'↑' if HIGHER_BETTER[m] else '↓'}")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "comparison_chart.png", dpi=150)
    plt.close()

    pd.set_option("display.width", 220, "display.max_columns", None)
    print("\n=========== SUMMARY (best method first) ===========")
    print(summary.to_string())
    print("\n=========== BEST METHOD PER METRIC ===========")
    print(best.to_string())
    print(f"\nOverall best method: {summary.index[0]}")
    print(f"All tables saved in: {SAVE_DIR}")


if __name__ == "__main__":
    main()