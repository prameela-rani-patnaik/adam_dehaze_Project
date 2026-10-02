"""
Hybrid single-image dehazing: Dark Channel Prior (He et al.) + Fattal (2008)
albedo/transmission decorrelation.

Haze model:  I(x) = J(x) t(x) + A (1 - t(x))

Pipeline
--------
1. DCP       -> atmospheric light A and a robust (but blocky/over-estimating) t_dcp.
2. Fattal    -> local transmission t_fat from the statistical independence of
                surface shading and transmission (detail preserving, but
                unreliable in flat / low-SNR regions).
3. Fusion    -> t_fat is calibrated to t_dcp's scale and blended with a
                per-pixel reliability weight (Fattal where reliable, DCP elsewhere).
4. Refine    -> guided filter, then radiance recovery with a lower bound t0.

Usage:
    Edit INPUT_PATH and OUTPUT_PATH near the bottom of this file, then run:
    python hybrid_dcp_fattal_dehaze.py
Requires: numpy, opencv-python
"""
import sys
import cv2
import numpy as np

EPS = 1e-6


# ----------------------------------------------------------------- utilities
def box(x, r):
    """Mean filter with window (2r+1)."""
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def local_cov(x, y, r):
    return box(x * y, r) - box(x, r) * box(y, r)


def guided_filter(guide, src, r, eps):
    """Gray-guide guided filter (He et al.)."""
    mean_g, mean_s = box(guide, r), box(src, r)
    var_g = box(guide * guide, r) - mean_g ** 2
    cov_gs = box(guide * src, r) - mean_g * mean_s
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return box(a, r) * guide + box(b, r)


# ------------------------------------------------------------------ DCP part
def dark_channel(img, patch=15):
    min_rgb = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(min_rgb, kernel)


def estimate_atmospheric_light(img, dark, top=0.001):
    h, w = dark.shape
    n = max(int(h * w * top), 1)
    idx = np.argpartition(dark.ravel(), -n)[-n:]          # haziest pixels
    cand = img.reshape(-1, 3)[idx]
    brightest = cand[np.argmax(cand.sum(axis=1))]         # pick brightest among them
    return brightest.astype(np.float32)


def transmission_dcp(img, A, omega=0.95, patch=15):
    return 1.0 - omega * dark_channel(img / A.reshape(1, 1, 3), patch)


# --------------------------------------------------------------- Fattal part
def transmission_fattal(img, A, r=3):
    """
    Fattal (2008): split each pixel into a component along the airlight
    direction (I_A) and a residual (I_R). Assuming surface albedo is
    uncorrelated with transmission in a local window, solve for eta
    (the albedo-related scalar) and obtain t.

    Returns t_fat (up to a scale) and a reliability map in [0, 1].
    """
    A_norm = np.linalg.norm(A) + EPS
    A_dir = A / A_norm

    I_A = img @ A_dir                                          # projection on A
    I_norm = np.linalg.norm(img, axis=2)
    I_R = np.sqrt(np.maximum(I_norm ** 2 - I_A ** 2, EPS))     # residual
    h = (I_norm - I_A) / (I_R + EPS)

    c_Ah = local_cov(I_A, h, r)
    c_Rh = local_cov(I_R, h, r)
    eta = c_Ah / (c_Rh + np.sign(c_Rh + EPS) * 1e-5)

    # stabilise eta: clip to sane range and smooth with a larger window
    eta = np.clip(eta, -2.0, 2.0)
    eta = box(eta, 4 * r)

    t = 1.0 - (I_A - eta * I_R) / A_norm

    # reliability: needs signal in I_R and a well-conditioned correlation
    std_R = np.sqrt(np.maximum(local_cov(I_R, I_R, r), 0))
    std_h = np.sqrt(np.maximum(local_cov(h, h, r), 0))
    corr = np.abs(c_Rh) / (std_R * std_h + EPS)
    rel = np.clip(std_R / 0.05, 0, 1) * np.clip(corr / 0.5, 0, 1)
    rel = box(rel, r)
    return t.astype(np.float32), rel.astype(np.float32)


# -------------------------------------------------------------------- hybrid
def fuse_transmission(t_dcp, t_fat, rel, max_fat_weight=0.7):
    """Calibrate Fattal's t to DCP scale, then blend by reliability."""
    haze_d = 1.0 - t_dcp
    haze_f = np.clip(1.0 - t_fat, 0, 1)

    # robust scale factor from reliable pixels
    mask = rel > 0.5
    if mask.sum() > 100:
        s = np.median(haze_d[mask]) / (np.median(haze_f[mask]) + EPS)
        s = float(np.clip(s, 0.3, 3.0))
    else:
        s = 1.0
    t_fat_cal = np.clip(1.0 - s * haze_f, 0.0, 1.0)

    w = max_fat_weight * rel
    return w * t_fat_cal + (1.0 - w) * t_dcp


def dehaze(img_bgr, patch=15, omega=0.95, t0=0.1, fattal_r=3,
           guide_r=40, guide_eps=1e-3):
    img = img_bgr.astype(np.float32) / 255.0

    dark = dark_channel(img, patch)
    A = estimate_atmospheric_light(img, dark)

    t_dcp = transmission_dcp(img, A, omega, patch)
    t_fat, rel = transmission_fattal(img, A, fattal_r)
    t = fuse_transmission(t_dcp, t_fat, rel)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = np.clip(guided_filter(gray, t.astype(np.float32), guide_r, guide_eps), t0, 1.0)

    J = (img - A.reshape(1, 1, 3)) / t[..., None] + A.reshape(1, 1, 3)
    J = np.clip(J, 0, 1)
    return (J * 255).astype(np.uint8), t, t_dcp, t_fat, A


# ----------------------------------------------------------------------- MAIN
# >>> SET YOUR PATHS HERE <<<
INPUT_PATH = r"C:\Users\patna\OneDrive\Desktop\IP_1\input_images\image1\Medium_haze.jpg"        # input hazy image
OUTPUT_PATH = r"C:\Users\patna\OneDrive\Desktop\IP_1\Hybrid_Results\DCP.Fattal_Results\image1\Medium_hazed.jpg"    # output (hazy | dehazed side by side)


def main():
    img = cv2.imread(INPUT_PATH)
    if img is None:
        sys.exit(f"Cannot read {INPUT_PATH}")

    J, t, t_dcp, t_fat, A = dehaze(img)
    print("Atmospheric light (BGR):", np.round(A, 3))

    def vis(m):
        return cv2.cvtColor((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    cv2.imwrite(OUTPUT_PATH, np.hstack([img, J]))
    cv2.imwrite(OUTPUT_PATH.rsplit(".", 1)[0] + "_transmission.png",
                np.hstack([vis(t_dcp), vis(t_fat), vis(t)]))
    print(f"Saved {OUTPUT_PATH} (left: hazy, right: dehazed) and transmission maps "
          f"(DCP | Fattal | Hybrid).")


if __name__ == "__main__":
    main()