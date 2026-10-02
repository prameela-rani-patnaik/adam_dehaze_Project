
"""
evaluation.py

Evaluate DCP, CAP and Fattal dehazing results.

Metrics:
    Full-reference:
        PSNR        Higher is better
        SSIM        Higher is better
        MSE         Lower is better
        MAE         Lower is better
        CIEDE2000    Lower is better

    No-reference:
        Entropy       Higher is better
        Contrast      Higher is better
        AvgGradient   Higher is better
        Colorfulness  Higher is better

Folder structure expected:

IP_1/
│
├── data/
│   └── organized_dataset/
│       ├── Low/
│       │   ├── hazy/
│       │   └── clear/
│       │
│       ├── Medium/
│       │   ├── hazy/
│       │   └── clear/
│       │
│       └── High/
│           ├── hazy/
│           └── clear/
│
├── Traditional_Results/
│   ├── DCP/
│   │   ├── Low/
│   │   ├── Medium/
│   │   └── High/
│   │
│   ├── CAP/
│   │   ├── Low/
│   │   ├── Medium/
│   │   └── High/
│   │
│   └── Fattal/
│       ├── Low/
│       ├── Medium/
│       └── High/
│
└── Traditional_Methods/
    └── evaluation.py


Run:

    python evaluation.py


Output:

Traditional_Results/
└── Evaluation/
    ├── Detailed.csv
    ├── Summary.csv
    ├── By_Haze.csv
    ├── Best_Method.csv
    ├── metrics_tables.xlsx
    └── comparison_chart.png
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage.color import (
    deltaE_ciede2000,
    rgb2lab
)

from skimage.metrics import (
    peak_signal_noise_ratio as psnr_fn,
    structural_similarity as ssim_fn
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\patna\OneDrive\Desktop\IP_1"
)

RESULTS_DIR = Path(r"C:\Users\patna\OneDrive\Desktop\IP_1\Traditional_Results"
    
)

DATASET_DIR = Path(r"C:\Users\patna\OneDrive\Desktop\IP_1\input_images\clear_image\image1.jpg"
  
)

SAVE_DIR = Path(
   r"C:\Users\patna\OneDrive\Desktop\IP_1\Traditional_Results\Evaluation_Results"
)


# ============================================================
# SETTINGS
# ============================================================

IMG_EXT = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

HAZE_LEVELS = [
    "Low",
    "Medium",
    "High"
]

METHODS = [
    "DCP_Results",
    "CAP_Results",
    "Fattal_Results"
]


# ============================================================
# WHICH DIRECTION IS BETTER?
# ============================================================

HIGHER_BETTER = {

    "PSNR": True,

    "SSIM": True,

    "MSE": False,

    "MAE": False,

    "CIEDE2000": False,

    "Entropy": True,

    "Contrast": True,

    "AvgGradient": True,

    "Colorfulness": True
}


# ============================================================
# READ IMAGE AS RGB
# ============================================================

def read_rgb(path):

    img = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if img is None:

        raise FileNotFoundError(
            f"Could not read image:\n{path}"
        )

    return cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# RESIZE IMAGE TO GROUND TRUTH
# ============================================================

def resize_to(img, reference):

    if img.shape[:2] != reference.shape[:2]:

        print(
            "      [info] resizing result "
            f"{img.shape[:2]} -> {reference.shape[:2]}"
        )

        img = cv2.resize(
            img,
            (
                reference.shape[1],
                reference.shape[0]
            ),
            interpolation=cv2.INTER_AREA
        )

    return img


# ============================================================
# FIND CLEAR GROUND-TRUTH IMAGE
# ============================================================

def find_ground_truth(
    result_path,
    haze_level
):

    clear_dir = (
        DATASET_DIR
        / haze_level
        / "clear"
    )

    if not clear_dir.exists():

        return None

    # --------------------------------------------------------
    # 1. Exact filename
    # --------------------------------------------------------

    exact = (
        clear_dir
        / result_path.name
    )

    if exact.exists():

        return exact

    # --------------------------------------------------------
    # 2. Same filename stem
    # --------------------------------------------------------

    for ext in IMG_EXT:

        candidate = (
            clear_dir
            / f"{result_path.stem}{ext}"
        )

        if candidate.exists():

            return candidate

    # --------------------------------------------------------
    # 3. Recursive search
    # --------------------------------------------------------

    for p in clear_dir.rglob("*"):

        if not p.is_file():

            continue

        if p.suffix.lower() not in IMG_EXT:

            continue

        if (
            p.stem.lower()
            == result_path.stem.lower()
        ):

            return p

    return None


# ============================================================
# FIND ORIGINAL HAZY IMAGE
# ============================================================

def find_hazy_image(
    result_path,
    haze_level
):

    hazy_dir = (
        DATASET_DIR
        / haze_level
        / "hazy"
    )

    if not hazy_dir.exists():

        return None

    # --------------------------------------------------------
    # 1. Exact filename
    # --------------------------------------------------------

    exact = (
        hazy_dir
        / result_path.name
    )

    if exact.exists():

        return exact

    # --------------------------------------------------------
    # 2. Same stem
    # --------------------------------------------------------

    for ext in IMG_EXT:

        candidate = (
            hazy_dir
            / f"{result_path.stem}{ext}"
        )

        if candidate.exists():

            return candidate

    # --------------------------------------------------------
    # 3. Recursive search
    # --------------------------------------------------------

    for p in hazy_dir.rglob("*"):

        if not p.is_file():

            continue

        if p.suffix.lower() not in IMG_EXT:

            continue

        if (
            p.stem.lower()
            == result_path.stem.lower()
        ):

            return p

    return None


# ============================================================
# FULL-REFERENCE METRICS
# ============================================================

def full_reference_metrics(
    result,
    ground_truth
):

    # --------------------------------------------------------
    # Match dimensions
    # --------------------------------------------------------

    result = resize_to(
        result,
        ground_truth
    )

    result_float = (
        result.astype(np.float64)
    )

    gt_float = (
        ground_truth.astype(np.float64)
    )

    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------

    mse = np.mean(
        (
            result_float
            - gt_float
        ) ** 2
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = np.mean(
        np.abs(
            result_float
            - gt_float
        )
    )

    # --------------------------------------------------------
    # PSNR
    # --------------------------------------------------------

    psnr = psnr_fn(
        ground_truth,
        result,
        data_range=255
    )

    # --------------------------------------------------------
    # SSIM
    # --------------------------------------------------------

    ssim = ssim_fn(
        ground_truth,
        result,
        channel_axis=2,
        data_range=255
    )

    # --------------------------------------------------------
    # CIEDE2000
    # --------------------------------------------------------

    gt_lab = rgb2lab(
        ground_truth
    )

    result_lab = rgb2lab(
        result
    )

    ciede2000 = np.mean(
        deltaE_ciede2000(
            gt_lab,
            result_lab
        )
    )

    return {

        "PSNR": float(psnr),

        "SSIM": float(ssim),

        "MSE": float(mse),

        "MAE": float(mae),

        "CIEDE2000": float(ciede2000)
    }


# ============================================================
# NO-REFERENCE METRICS
# ============================================================

def no_reference_metrics(img):

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    gray_float = (
        gray.astype(np.float64)
    )

    # ========================================================
    # ENTROPY
    # ========================================================

    histogram = np.bincount(
        gray.ravel(),
        minlength=256
    )

    histogram = (
        histogram
        / gray.size
    )

    histogram = histogram[
        histogram > 0
    ]

    entropy = -np.sum(
        histogram
        * np.log2(histogram)
    )

    # ========================================================
    # CONTRAST
    # ========================================================

    contrast = gray_float.std()

    # ========================================================
    # AVERAGE GRADIENT
    # ========================================================

    gx = cv2.Sobel(
        gray_float,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray_float,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    avg_gradient = np.mean(
        np.sqrt(
            (
                gx ** 2
                + gy ** 2
            )
            / 2
        )
    )

    # ========================================================
    # COLORFULNESS
    # ========================================================

    r = img[..., 0].astype(
        np.float64
    )

    g = img[..., 1].astype(
        np.float64
    )

    b = img[..., 2].astype(
        np.float64
    )

    rg = r - g

    yb = (
        0.5 * (r + g)
        - b
    )

    colorfulness = (

        np.sqrt(
            rg.std() ** 2
            + yb.std() ** 2
        )

        +

        0.3
        * np.sqrt(
            rg.mean() ** 2
            + yb.mean() ** 2
        )
    )

    return {

        "Entropy": float(entropy),

        "Contrast": float(contrast),

        "AvgGradient": float(
            avg_gradient
        ),

        "Colorfulness": float(
            colorfulness
        )
    }


# ============================================================
# FIND ALL RESULT IMAGES
# ============================================================

def get_result_images(
    method_dir
):

    files = []

    if not method_dir.exists():

        return files

    # --------------------------------------------------------
    # Search Low / Medium / High
    # --------------------------------------------------------

    for haze_dir in method_dir.iterdir():

        if not haze_dir.is_dir():

            continue

        if (
            haze_dir.name.lower()
            not in [
                "low",
                "medium",
                "high"
            ]
        ):

            continue

        for p in haze_dir.rglob("*"):

            if not p.is_file():

                continue

            if (
                p.suffix.lower()
                not in IMG_EXT
            ):

                continue

            # Ignore transmission maps
            if (
                "transmission"
                in p.stem.lower()
            ):

                continue

            files.append(p)

    return sorted(files)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Create evaluation directory
    # --------------------------------------------------------

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRADITIONAL DEHAZING EVALUATION")
    print("=" * 70)

    print(
        f"Project Root : {PROJECT_ROOT}"
    )

    print(
        f"Results Dir  : {RESULTS_DIR}"
    )

    print(
        f"Dataset Dir  : {DATASET_DIR}"
    )

    print(
        f"Save Dir     : {SAVE_DIR}"
    )

    print()

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not RESULTS_DIR.exists():

        raise SystemExit(
            "\nERROR: Traditional_Results folder "
            "does not exist:\n"
            f"{RESULTS_DIR}"
        )

    if not DATASET_DIR.exists():

        raise SystemExit(
            "\nERROR: Dataset folder "
            "does not exist:\n"
            f"{DATASET_DIR}"
        )

    # --------------------------------------------------------
    # Store all rows
    # --------------------------------------------------------

    all_rows = []

    # ========================================================
    # PROCESS EACH METHOD
    # ========================================================

    for method in METHODS:

        print()
        print("=" * 70)
        print(f"METHOD: {method}")
        print("=" * 70)

        method_dir = (
            RESULTS_DIR
            / method
        )

        # ----------------------------------------------------
        # Check method folder
        # ----------------------------------------------------

        if not method_dir.exists():

            print(
                f"[WARNING] Folder not found:"
            )

            print(
                f"          {method_dir}"
            )

            continue

        # ----------------------------------------------------
        # Find images
        # ----------------------------------------------------

        result_files = get_result_images(
            method_dir
        )

        print(
            f"Images found: {len(result_files)}"
        )

        if not result_files:

            print(
                "[WARNING] No result images found."
            )

            continue

        # ====================================================
        # PROCESS EACH RESULT IMAGE
        # ====================================================

        for result_path in result_files:

            # ------------------------------------------------
            # Determine haze level
            # ------------------------------------------------

            haze_level = (
                result_path.parent.name
            )

            if (
                haze_level.lower()
                not in [
                    "low",
                    "medium",
                    "high"
                ]
            ):

                print(
                    f"[SKIP] Unknown haze level: "
                    f"{result_path}"
                )

                continue

            # ------------------------------------------------
            # Read result image
            # ------------------------------------------------

            try:

                result_img = read_rgb(
                    result_path
                )

            except Exception as e:

                print(
                    f"[ERROR] Cannot read "
                    f"{result_path.name}: {e}"
                )

                continue

            # ------------------------------------------------
            # Find ground truth
            # ------------------------------------------------

            gt_path = find_ground_truth(
                result_path,
                haze_level
            )

            if gt_path is None:

                print(
                    f"[WARNING] Ground truth not found:"
                )

                print(
                    f"          Result: {result_path.name}"
                )

                print(
                    f"          Haze:   {haze_level}"
                )

                continue

            # ------------------------------------------------
            # Read ground truth
            # ------------------------------------------------

            try:

                gt_img = read_rgb(
                    gt_path
                )

            except Exception as e:

                print(
                    f"[ERROR] Cannot read GT "
                    f"{gt_path.name}: {e}"
                )

                continue

            # ------------------------------------------------
            # Calculate full-reference metrics
            # ------------------------------------------------

            try:

                ref_metrics = (
                    full_reference_metrics(
                        result_img,
                        gt_img
                    )
                )

            except Exception as e:

                print(
                    f"[ERROR] Metrics failed "
                    f"for {result_path.name}: {e}"
                )

                continue

            # ------------------------------------------------
            # Calculate no-reference metrics
            # ------------------------------------------------

            try:

                nr_metrics = (
                    no_reference_metrics(
                        result_img
                    )
                )

            except Exception as e:

                print(
                    f"[ERROR] No-reference metrics "
                    f"failed for {result_path.name}: {e}"
                )

                continue

            # ------------------------------------------------
            # Create row
            # ------------------------------------------------

            row = {

                "Method": method,

                "Image": result_path.stem,

                "Haze": haze_level,

                "Result_File": result_path.name,

                "GT_File": gt_path.name
            }

            # Add metrics
            row.update(
                ref_metrics
            )

            row.update(
                nr_metrics
            )

            # =================================================
            # COMPARE WITH ORIGINAL HAZY IMAGE
            # =================================================

            hazy_path = find_hazy_image(
                result_path,
                haze_level
            )

            if hazy_path is not None:

                try:

                    hazy_img = read_rgb(
                        hazy_path
                    )

                    hazy_metrics = (
                        full_reference_metrics(
                            hazy_img,
                            gt_img
                        )
                    )

                    row["PSNR_gain"] = (
                        row["PSNR"]
                        - hazy_metrics["PSNR"]
                    )

                    row["SSIM_gain"] = (
                        row["SSIM"]
                        - hazy_metrics["SSIM"]
                    )

                except Exception:

                    row["PSNR_gain"] = np.nan

                    row["SSIM_gain"] = np.nan

            else:

                row["PSNR_gain"] = np.nan

                row["SSIM_gain"] = np.nan

            # ------------------------------------------------
            # Store row
            # ------------------------------------------------

            all_rows.append(
                row
            )

            print(
                f"[OK] "
                f"{method:<8} | "
                f"{haze_level:<7} | "
                f"{result_path.name}"
            )

    # ========================================================
    # CHECK IF ANY IMAGE WAS EVALUATED
    # ========================================================

    if not all_rows:

        raise SystemExit(
            "\n"
            "==================================================\n"
            "ERROR: No images were successfully evaluated.\n"
            "==================================================\n"
            "\n"
            "Check that:\n"
            "1. DCP/CAP/Fattal result images exist.\n"
            "2. They are inside Low/Medium/High folders.\n"
            "3. Corresponding clear images exist.\n"
            "4. Result and clear images have matching names.\n"
        )

    # ========================================================
    # DETAILED DATAFRAME
    # ========================================================

    detailed = pd.DataFrame(
        all_rows
    )

    detailed = detailed.sort_values(
        [
            "Method",
            "Haze",
            "Image"
        ]
    )

    detailed = detailed.round(4)

    # ========================================================
    # METRIC LIST
    # ========================================================

    metric_columns = [

        "PSNR",

        "SSIM",

        "MSE",

        "MAE",

        "CIEDE2000",

        "Entropy",

        "Contrast",

        "AvgGradient",

        "Colorfulness",

        "PSNR_gain",

        "SSIM_gain"
    ]

    existing_metrics = [

        metric
        for metric in metric_columns
        if metric in detailed.columns
    ]

    # ========================================================
    # SUMMARY BY METHOD
    # ========================================================

    summary = (
        detailed
        .groupby("Method")
        [existing_metrics]
        .mean()
    )

    # ========================================================
    # RANK METHODS
    # ========================================================

    ranking_metrics = [

        metric
        for metric in HIGHER_BETTER
        if metric in summary.columns
    ]

    ranks = pd.DataFrame(
        index=summary.index
    )

    for metric in ranking_metrics:

        ranks[metric] = (
            summary[metric]
            .rank(
                ascending=(
                    not HIGHER_BETTER[metric]
                ),
                method="min"
            )
        )

    summary["Average_Rank"] = (
        ranks.mean(axis=1)
    )

    summary = summary.sort_values(
        "Average_Rank"
    )

    summary = summary.round(4)

    # ========================================================
    # RESULTS BY HAZE LEVEL
    # ========================================================

    by_haze = (
        detailed
        .groupby(
            [
                "Method",
                "Haze"
            ]
        )[existing_metrics]
        .mean()
    )

    by_haze = by_haze.round(4)

    # ========================================================
    # BEST METHOD PER METRIC
    # ========================================================

    best_rows = []

    for metric in ranking_metrics:

        if HIGHER_BETTER[metric]:

            best_method = (
                summary[metric]
                .idxmax()
            )

            best_value = (
                summary[metric]
                .max()
            )

            rule = (
                "higher is better"
            )

        else:

            best_method = (
                summary[metric]
                .idxmin()
            )

            best_value = (
                summary[metric]
                .min()
            )

            rule = (
                "lower is better"
            )

        best_rows.append({

            "Metric": metric,

            "Best_Method": best_method,

            "Best_Value": best_value,

            "Rule": rule
        })

    best = pd.DataFrame(
        best_rows
    )

    best = best.set_index(
        "Metric"
    )

    best = best.round(4)

    # ========================================================
    # SAVE DETAILED CSV
    # ========================================================

    detailed.to_csv(
        SAVE_DIR
        / "Detailed.csv",
        index=False
    )

    # ========================================================
    # SAVE SUMMARY CSV
    # ========================================================

    summary.to_csv(
        SAVE_DIR
        / "Summary.csv"
    )

    # ========================================================
    # SAVE BY-HAZE CSV
    # ========================================================

    by_haze.to_csv(
        SAVE_DIR
        / "By_Haze.csv"
    )

    # ========================================================
    # SAVE BEST METHOD CSV
    # ========================================================

    best.to_csv(
        SAVE_DIR
        / "Best_Method.csv"
    )

    # ========================================================
    # SAVE EXCEL
    # ========================================================

    excel_path = (
        SAVE_DIR
        / "metrics_tables.xlsx"
    )

    try:

        with pd.ExcelWriter(
            excel_path
        ) as writer:

            detailed.to_excel(
                writer,
                sheet_name="Detailed",
                index=False
            )

            summary.to_excel(
                writer,
                sheet_name="Summary"
            )

            by_haze.to_excel(
                writer,
                sheet_name="By_Haze"
            )

            best.to_excel(
                writer,
                sheet_name="Best_Method"
            )

        print()
        print(
            f"Excel saved: {excel_path}"
        )

    except Exception as e:

        print()
        print(
            "[WARNING] Could not create Excel file."
        )

        print(
            f"Reason: {e}"
        )

        print(
            "CSV files were still saved."
        )

    # ========================================================
    # CREATE COMPARISON CHART
    # ========================================================

    chart_metrics = [

        "PSNR",

        "SSIM",

        "MSE",

        "MAE",

        "CIEDE2000",

        "Entropy",

        "Contrast",

        "AvgGradient",

        "Colorfulness"
    ]

    chart_metrics = [

        metric
        for metric in chart_metrics
        if metric in summary.columns
    ]

    if chart_metrics:

        number_of_metrics = (
            len(chart_metrics)
        )

        columns = 3

        number_of_rows = (
            number_of_metrics
            + columns
            - 1
        ) // columns

        fig, axes = plt.subplots(
            number_of_rows,
            columns,
            figsize=(
                5 * columns,
                4 * number_of_rows
            )
        )

        axes = np.atleast_1d(
            axes
        ).ravel()

        for ax, metric in zip(
            axes,
            chart_metrics
        ):

            summary[metric].plot(
                kind="bar",
                ax=ax,
                edgecolor="black"
            )

            arrow = (
                "↑"
                if HIGHER_BETTER[metric]
                else "↓"
            )

            ax.set_title(
                f"{metric} {arrow}"
            )

            ax.set_xlabel(
                "Method"
            )

            ax.set_ylabel(
                metric
            )

            ax.tick_params(
                axis="x",
                rotation=30
            )

        # Hide unused plots
        for ax in axes[
            len(chart_metrics):
        ]:

            ax.axis("off")

        plt.tight_layout()

        chart_path = (
            SAVE_DIR
            / "comparison_chart.png"
        )

        plt.savefig(
            chart_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Chart saved: {chart_path}"
        )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    pd.set_option(
        "display.width",
        250
    )

    pd.set_option(
        "display.max_columns",
        None
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Successfully evaluated: "
        f"{len(detailed)} images"
    )

    print()
    print("=========== SUMMARY ===========")

    print(
        summary.to_string()
    )

    print()
    print(
        "=========== BEST METHOD PER METRIC ==========="
    )

    print(
        best.to_string()
    )

    print()
    print(
        "=========== RESULTS BY HAZE LEVEL ==========="
    )

    print(
        by_haze.to_string()
    )

    print()
    print(
        "=========== OUTPUT FILES ==========="
    )

    print(
        SAVE_DIR
        / "Detailed.csv"
    )

    print(
        SAVE_DIR
        / "Summary.csv"
    )

    print(
        SAVE_DIR
        / "By_Haze.csv"
    )

    print(
        SAVE_DIR
        / "Best_Method.csv"
    )

    print(
        SAVE_DIR
        / "metrics_tables.xlsx"
    )

    print(
        SAVE_DIR
        / "comparison_chart.png"
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()