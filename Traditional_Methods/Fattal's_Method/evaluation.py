import cv2
import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
    mean_squared_error
)

# ============================================================
# IMAGE PATHS
# ============================================================

clear_path = r"C:\Users\patna\OneDrive\Desktop\IP_1\dehazing_using_color_lines-master\images\8914_0.85_0.16.jpg"

dehazed_path = r"C:\Users\patna\OneDrive\Desktop\IP_1\dehazing_using_color_lines-master\images\8914_0.85_0.16_dehazed.jpg"


# ============================================================
# READ IMAGES
# ============================================================

clear = cv2.imread(clear_path)
dehazed = cv2.imread(dehazed_path)

if clear is None:
    raise FileNotFoundError("Clear image not found")

if dehazed is None:
    raise FileNotFoundError("Dehazed image not found")


# ============================================================
# MAKE SAME SIZE
# ============================================================

if clear.shape != dehazed.shape:
    dehazed = cv2.resize(
        dehazed,
        (clear.shape[1], clear.shape[0])
    )


# ============================================================
# CONVERT BGR → RGB
# ============================================================

clear = cv2.cvtColor(clear, cv2.COLOR_BGR2RGB)
dehazed = cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB)


# ============================================================
# NORMALIZE
# ============================================================

clear = clear.astype(np.float32) / 255.0
dehazed = dehazed.astype(np.float32) / 255.0


# ============================================================
# MSE
# ============================================================

mse = mean_squared_error(clear, dehazed)


# ============================================================
# MAE
# ============================================================

mae = np.mean(np.abs(clear - dehazed))


# ============================================================
# PSNR
# ============================================================

psnr = peak_signal_noise_ratio(
    clear,
    dehazed,
    data_range=1.0
)


# ============================================================
# SSIM
# ============================================================

ssim = structural_similarity(
    clear,
    dehazed,
    channel_axis=2,
    data_range=1.0
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n======================================")
print("   COLOR LINES: CLEAR vs DEHAZED")
print("======================================")

print(f"PSNR : {psnr:.4f} dB")
print(f"SSIM : {ssim:.4f}")
print(f"MSE  : {mse:.6f}")
print(f"MAE  : {mae:.6f}")

print("======================================")