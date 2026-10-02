"""
Hybrid single-image dehazing: Color Attenuation Prior (Zhu et al.) +
Fattal (2008) albedo/transmission decorrelation.

Haze model:  I(x) = J(x) t(x) + A (1 - t(x))

Pipeline
--------
1. CAP    -> depth d = theta0 + theta1*V + theta2*S, t_cap = exp(-beta*d).
             Atmospheric light A is taken from the farthest (deepest) pixels.
             Robust and smooth, but coarse in fine detail.
2. Fattal -> local transmission t_fat from the statistical independence of
             surface shading and transmission. Detail preserving, but only
             defined up to a scale and unreliable in flat / low-SNR regions.
3. Fusion -> t_fat is calibrated to t_cap's scale, then blended per pixel
             with a reliability weight (Fattal where reliable, CAP elsewhere).
4. Refine -> guided filter, then radiance recovery with lower bound t0.

Requires: numpy, opencv-python
"""
import sys
import cv2
import numpy as np

# >>> SET YOUR PATHS HERE <<<
INPUT_PATH = r"C:\Users\patna\OneDrive\Desktop\IP_1\input_images\image1\low_haze.jpg"        # input hazy image
OUTPUT_PATH = r"C:\Users\patna\OneDrive\Desktop\IP_1\Hybrid_Results\CAP.Fattal_Results\image1\low_haze_dehazed.jpg"     # output (hazy | dehazed side by side)

EPS = 1e-6


# ----------------------------------------------------------------- utilities
def box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def local_cov(x, y, r):
    return box(x * y, r) - box(x, r) * box(y, r)


def min_filter(x, patch):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(x, k)


def guided_filter(guide, src, r, eps):
    mean_g, mean_s = box(guide, r), box(src, r)
    var_g = box(guide * guide, r) - mean_g ** 2
    cov_gs = box(guide * src, r) - mean_g * mean_s
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return box(a, r) * guide + box(b, r)


# ------------------------------------------------------------------ CAP part
def cap_depth(img, theta=(0.1213, 0.9597, -0.7800), patch=15):
    """Depth from brightness/saturation (default theta from Zhu et al. 2015)."""
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    s, v = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    depth = theta[0] + theta[1] * v + theta[2] * s
    depth = min_filter(depth.astype(np.float32), patch)      # suppress white-object noise
    return box(depth, patch // 2)


def estimate_atmospheric_light(img, depth, top=0.001):
    """Brightest pixel among the top 0.1% deepest pixels."""
    h, w = depth.shape
    n = max(int(h * w * top), 1)
    idx = np.argpartition(depth.ravel(), -n)[-n:]
    cand = img.reshape(-1, 3)[idx]
    return cand[np.argmax(cand.sum(axis=1))].astype(np.float32)


def transmission_cap(depth, beta=1.0):
    return np.exp(-beta * depth).astype(np.float32)


# --------------------------------------------------------------- Fattal part
def transmission_fattal(img, A, r=3):
    """
    Split each pixel into a component along the airlight direction (I_A) and a
    residual (I_R). Assuming albedo is locally uncorrelated with transmission,
    solve for eta and obtain t (up to scale). Also returns a reliability map.
    """
    A_norm = np.linalg.norm(A) + EPS
    A_dir = A / A_norm

    I_A = img @ A_dir
    I_norm = np.linalg.norm(img, axis=2)
    I_R = np.sqrt(np.maximum(I_norm ** 2 - I_A ** 2, EPS))
    h = (I_norm - I_A) / (I_R + EPS)

    c_Ah = local_cov(I_A, h, r)
    c_Rh = local_cov(I_R, h, r)
    eta = c_Ah / (c_Rh + np.sign(c_Rh + EPS) * 1e-5)
    eta = box(np.clip(eta, -2.0, 2.0), 4 * r)                # stabilise

    t = 1.0 - (I_A - eta * I_R) / A_norm

    std_R = np.sqrt(np.maximum(local_cov(I_R, I_R, r), 0))
    std_h = np.sqrt(np.maximum(local_cov(h, h, r), 0))
    corr = np.abs(c_Rh) / (std_R * std_h + EPS)
    rel = box(np.clip(std_R / 0.05, 0, 1) * np.clip(corr / 0.5, 0, 1), r)
    return t.astype(np.float32), rel.astype(np.float32)


# -------------------------------------------------------------------- hybrid
def fuse_transmission(t_cap, t_fat, rel, max_fat_weight=0.7):
    """Calibrate Fattal's t to CAP's scale, then blend by reliability."""
    haze_c = 1.0 - t_cap
    haze_f = np.clip(1.0 - t_fat, 0, 1)

    mask = rel > 0.5
    if mask.sum() > 100:
        s = np.median(haze_c[mask]) / (np.median(haze_f[mask]) + EPS)
        s = float(np.clip(s, 0.3, 3.0))
    else:
        s = 1.0
    t_fat_cal = np.clip(1.0 - s * haze_f, 0.0, 1.0)

    w = max_fat_weight * rel
    return w * t_fat_cal + (1.0 - w) * t_cap


def dehaze(img_bgr, patch=15, beta=1.0, t0=0.1, fattal_r=3,
           guide_r=40, guide_eps=1e-3):
    img = img_bgr.astype(np.float32) / 255.0

    depth = cap_depth(img, patch=patch)
    A = estimate_atmospheric_light(img, depth)

    t_cap = transmission_cap(depth, beta)
    t_fat, rel = transmission_fattal(img, A, fattal_r)
    t = fuse_transmission(t_cap, t_fat, rel)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = np.clip(guided_filter(gray, t.astype(np.float32), guide_r, guide_eps), t0, 1.0)

    J = (img - A.reshape(1, 1, 3)) / t[..., None] + A.reshape(1, 1, 3)
    J = np.clip(J, 0, 1)
    return (J * 255).astype(np.uint8), t, t_cap, t_fat, A


# ---------------------------------------------------------------------- main
def main():
    img = cv2.imread(INPUT_PATH)
    if img is None:
        sys.exit(f"Cannot read {INPUT_PATH}")

    J, t, t_cap, t_fat, A = dehaze(img)
    print("Atmospheric light (BGR):", np.round(A, 3))

    def vis(m):
        return cv2.cvtColor((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    cv2.imwrite(OUTPUT_PATH, np.hstack([img, J]))
    cv2.imwrite(OUTPUT_PATH.rsplit(".", 1)[0] + "_transmission.png",
                np.hstack([vis(t_cap), vis(t_fat), vis(t)]))
    print(f"Saved {OUTPUT_PATH} (left: hazy, right: dehazed) and transmission maps "
          f"(CAP | Fattal | Hybrid).")


if __name__ == "__main__":
    main()