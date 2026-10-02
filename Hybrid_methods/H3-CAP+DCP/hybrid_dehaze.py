"""
Hybrid single-image dehazing: Dark Channel Prior (He et al.) + Color
Attenuation Prior (Zhu et al.).

Haze model:  I(x) = J(x) t(x) + A (1 - t(x))

Pipeline
--------
1. DCP -> atmospheric light A and transmission t_dcp.
          Strong on textured/colorful scenes, but fails on bright regions
          (sky, white objects) where the dark channel prior is violated.
2. CAP -> scene depth from brightness (V) and saturation (S):
              d = theta0 + theta1 * V + theta2 * S
          then t_cap = exp(-beta * d). Robust on sky / bright regions,
          but can be less accurate in fine details.
3. Fusion -> per-pixel blend: CAP gets more weight where the DCP is
          unreliable (high dark channel), DCP elsewhere.
4. Refine -> guided filter, then radiance recovery with lower bound t0.

Requires: numpy, opencv-python
"""
import sys
import cv2
import numpy as np

# >>> SET YOUR PATHS HERE <<<
INPUT_PATH = r"C:\Users\patna\OneDrive\Desktop\IP_1\input_images\image1\Medium_haze.jpg" \
""        # input hazy image
OUTPUT_PATH = r"C:\Users\patna\OneDrive\Desktop\IP_1\Hybrid_Results\DCP.CAP_Results\image1\Medium_haze_dehazed.jpg"     # output (hazy | dehazed side by side)


# ----------------------------------------------------------------- utilities
def box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide, src, r, eps):
    mean_g, mean_s = box(guide, r), box(src, r)
    var_g = box(guide * guide, r) - mean_g ** 2
    cov_gs = box(guide * src, r) - mean_g * mean_s
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return box(a, r) * guide + box(b, r)


def min_filter(x, patch):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(x, k)


# ------------------------------------------------------------------ DCP part
def dark_channel(img, patch=15):
    return min_filter(np.min(img, axis=2), patch)


def estimate_atmospheric_light(img, dark, top=0.001):
    h, w = dark.shape
    n = max(int(h * w * top), 1)
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    cand = img.reshape(-1, 3)[idx]
    return cand[np.argmax(cand.sum(axis=1))].astype(np.float32)


def transmission_dcp(img, A, omega=0.95, patch=15):
    dark = dark_channel(img / A.reshape(1, 1, 3), patch)
    return 1.0 - omega * dark, dark


# ------------------------------------------------------------------ CAP part
def transmission_cap(img_bgr, theta=(0.1213, 0.9597, -0.7800), beta=1.0, patch=15):
    """
    Color Attenuation Prior: depth is positively correlated with the
    brightness-saturation difference. Default theta are the values learned
    by Zhu et al. (2015).
    """
    hsv = cv2.cvtColor((img_bgr * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    s, v = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    depth = theta[0] + theta[1] * v + theta[2] * s
    depth = min_filter(depth.astype(np.float32), patch)      # remove white-object noise
    depth = box(depth, patch // 2)                           # light smoothing
    return np.exp(-beta * depth).astype(np.float32), depth


# -------------------------------------------------------------------- hybrid
def fuse_transmission(t_dcp, t_cap, dark_norm, base_cap_weight=0.3):
    """
    DCP is unreliable where the normalised dark channel is high (sky, white
    objects). Smoothly shift weight towards CAP in those regions.
    """
    u = np.clip((dark_norm - 0.5) / 0.3, 0.0, 1.0)           # 0 = DCP reliable, 1 = not
    u = u * u * (3 - 2 * u)                                  # smoothstep
    w_cap = base_cap_weight + (1.0 - base_cap_weight) * u
    return w_cap * t_cap + (1.0 - w_cap) * t_dcp


def dehaze(img_bgr, patch=15, omega=0.95, t0=0.1, beta=1.0,
           guide_r=40, guide_eps=1e-3):
    img = img_bgr.astype(np.float32) / 255.0

    dark = dark_channel(img, patch)
    A = estimate_atmospheric_light(img, dark)

    t_dcp, dark_norm = transmission_dcp(img, A, omega, patch)
    t_cap, _ = transmission_cap(img, beta=beta, patch=patch)
    t = fuse_transmission(t_dcp, t_cap, dark_norm)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = np.clip(guided_filter(gray, t.astype(np.float32), guide_r, guide_eps), t0, 1.0)

    J = (img - A.reshape(1, 1, 3)) / t[..., None] + A.reshape(1, 1, 3)
    J = np.clip(J, 0, 1)
    return (J * 255).astype(np.uint8), t, t_dcp, t_cap, A


# ---------------------------------------------------------------------- main
def main():
    img = cv2.imread(INPUT_PATH)
    if img is None:
        sys.exit(f"Cannot read {INPUT_PATH}")

    J, t, t_dcp, t_cap, A = dehaze(img)
    print("Atmospheric light (BGR):", np.round(A, 3))

    def vis(m):
        return cv2.cvtColor((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    cv2.imwrite(OUTPUT_PATH, np.hstack([img, J]))
    cv2.imwrite(OUTPUT_PATH.rsplit(".", 1)[0] + "_transmission.png",
                np.hstack([vis(t_dcp), vis(t_cap), vis(t)]))
    print(f"Saved {OUTPUT_PATH} (left: hazy, right: dehazed) and transmission maps "
          f"(DCP | CAP | Hybrid).")


if __name__ == "__main__":
    main()