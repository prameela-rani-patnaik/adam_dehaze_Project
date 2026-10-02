import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
    mean_squared_error
)


# ============================================================
# PATHS
# ============================================================

clear_path = (
   r"C:\Users\patna\OneDrive\Desktop\IP_1\data\organized_dataset\High\clear\8961.jpg"
)

hazy_path = (
     r"C:\Users\patna\OneDrive\Desktop\IP_1\data\organized_dataset\Medium\hazy\8961_1_0.12.jpg"
)

dehazed_path = (
    r"C:\Users\patna\OneDrive\Desktop\IP_1\DCP_results\dehazed_outputs\image_2\dehazed_output_medium(2).jpg"
)
results_folder = (
    r"C:\Users\patna\OneDrive\Desktop\IP_1"
    r"\data\results"
)

os.makedirs(results_folder, exist_ok=True)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    image = cv2.imread(path)

    if image is None:
        print(f"\nERROR: Could not load image:")
        print(path)
        return None

    return image


# ============================================================
# ENTROPY
# ============================================================

def calculate_entropy(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    histogram = histogram.ravel()

    probability = histogram / np.sum(histogram)

    probability = probability[probability > 0]

    entropy = -np.sum(
        probability * np.log2(probability)
    )

    return entropy


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(reference, image):

    reference_gray = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )

    image_gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # PSNR
    psnr = peak_signal_noise_ratio(
        reference_gray,
        image_gray,
        data_range=255
    )

    # SSIM
    ssim = structural_similarity(
        reference_gray,
        image_gray,
        data_range=255
    )

    # MSE
    mse = mean_squared_error(
        reference_gray,
        image_gray
    )

    # MAE
    mae = np.mean(
        np.abs(
            reference_gray.astype(np.float32)
            -
            image_gray.astype(np.float32)
        )
    )

    # Entropy
    entropy = calculate_entropy(image)

    return psnr, ssim, mse, mae, entropy


# ============================================================
# LOAD ALL IMAGES
# ============================================================

print("\nLoading images...")

clear_image = load_image(clear_path)
hazy_image = load_image(hazy_path)
dehazed_image = load_image(dehazed_path)


if clear_image is None or hazy_image is None or dehazed_image is None:
    print("\nProgram stopped because an image could not be loaded.")
    exit()


print(f"Clear image   : Loaded")
print(f"Hazy image    : Loaded")
print(f"Dehazed image : Loaded")


# ============================================================
# CLEAR IMAGE ENTROPY
# ============================================================

clear_entropy = calculate_entropy(clear_image)


print("\n")
print("=" * 60)
print("CLEAR IMAGE METRICS")
print("=" * 60)

print(f"Entropy : {clear_entropy:.4f}")

print("=" * 60)


# ============================================================
# HAZY VS CLEAR
# ============================================================

hazy_psnr, hazy_ssim, hazy_mse, hazy_mae, hazy_entropy = \
    calculate_metrics(clear_image, hazy_image)


print("\n")
print("=" * 60)
print("HAZY IMAGE vs CLEAR IMAGE")
print("=" * 60)

print(f"PSNR    : {hazy_psnr:.4f}")
print(f"SSIM    : {hazy_ssim:.4f}")
print(f"MSE     : {hazy_mse:.4f}")
print(f"MAE     : {hazy_mae:.4f}")
print(f"Entropy : {hazy_entropy:.4f}")

print("=" * 60)


# ============================================================
# DEHAZED VS CLEAR
# ============================================================

dehazed_psnr, dehazed_ssim, dehazed_mse, dehazed_mae, dehazed_entropy = \
    calculate_metrics(clear_image, dehazed_image)


print("\n")
print("=" * 60)
print("DEHAZED IMAGE vs CLEAR IMAGE")
print("=" * 60)

print(f"PSNR    : {dehazed_psnr:.4f}")
print(f"SSIM    : {dehazed_ssim:.4f}")
print(f"MSE     : {dehazed_mse:.4f}")
print(f"MAE     : {dehazed_mae:.4f}")
print(f"Entropy : {dehazed_entropy:.4f}")

print("=" * 60)


# ============================================================
# IMPROVEMENT
# ============================================================

psnr_improvement = dehazed_psnr - hazy_psnr
ssim_improvement = dehazed_ssim - hazy_ssim
mse_change = dehazed_mse - hazy_mse
mae_change = dehazed_mae - hazy_mae
entropy_change = dehazed_entropy - hazy_entropy


print("\n")
print("=" * 60)
print("IMPROVEMENT AFTER DEHAZING")
print("=" * 60)

print(f"PSNR improvement : {psnr_improvement:+.4f} dB")
print(f"SSIM improvement : {ssim_improvement:+.4f}")
print(f"MSE change       : {mse_change:+.4f}")
print(f"MAE change       : {mae_change:+.4f}")
print(f"Entropy change   : {entropy_change:+.4f}")

print("=" * 60)


# ============================================================
# FINAL TABLE
# ============================================================

print("\n")
print("=" * 75)
print("FINAL IMAGE QUALITY COMPARISON")
print("=" * 75)

print(
    f"{'Image':<15}"
    f"{'PSNR':>12}"
    f"{'SSIM':>12}"
    f"{'MSE':>15}"
    f"{'MAE':>12}"
    f"{'Entropy':>12}"
)

print("-" * 75)

print(
    f"{'Clear':<15}"
    f"{'--':>12}"
    f"{'--':>12}"
    f"{'--':>15}"
    f"{'--':>12}"
    f"{clear_entropy:>12.4f}"
)

print(
    f"{'Hazy':<15}"
    f"{hazy_psnr:>12.4f}"
    f"{hazy_ssim:>12.4f}"
    f"{hazy_mse:>15.4f}"
    f"{hazy_mae:>12.4f}"
    f"{hazy_entropy:>12.4f}"
)

print(
    f"{'Dehazed':<15}"
    f"{dehazed_psnr:>12.4f}"
    f"{dehazed_ssim:>12.4f}"
    f"{dehazed_mse:>15.4f}"
    f"{dehazed_mae:>12.4f}"
    f"{dehazed_entropy:>12.4f}"
)

print("=" * 75)


# ============================================================
# VISUAL COMPARISON
# ============================================================

clear_rgb = cv2.cvtColor(
    clear_image,
    cv2.COLOR_BGR2RGB
)

hazy_rgb = cv2.cvtColor(
    hazy_image,
    cv2.COLOR_BGR2RGB
)

dehazed_rgb = cv2.cvtColor(
    dehazed_image,
    cv2.COLOR_BGR2RGB
)


plt.figure(figsize=(18, 6))


# ------------------------------------------------------------
# CLEAR
# ------------------------------------------------------------

plt.subplot(1, 3, 1)

plt.imshow(clear_rgb)

plt.title(
    f"CLEAR IMAGE\n"
    f"Ground Truth | Entropy: {clear_entropy:.4f}",
    fontsize=13
)

plt.axis("off")


# ------------------------------------------------------------
# HAZY
# ------------------------------------------------------------

plt.subplot(1, 3, 2)

plt.imshow(hazy_rgb)

plt.title(
    f"HAZY IMAGE\n"
    f"PSNR: {hazy_psnr:.2f} dB | "
    f"SSIM: {hazy_ssim:.4f}",
    fontsize=13
)

plt.axis("off")


# ------------------------------------------------------------
# DEHAZED
# ------------------------------------------------------------

plt.subplot(1, 3, 3)

plt.imshow(dehazed_rgb)

plt.title(
    f"DEHAZED IMAGE (DCP)\n"
    f"PSNR: {dehazed_psnr:.2f} dB | "
    f"SSIM: {dehazed_ssim:.4f}",
    fontsize=13
)

plt.axis("off")


plt.tight_layout()


# ============================================================
# SAVE VISUAL COMPARISON
# ============================================================

visual_path = os.path.join(
    results_folder,
    "visual_comparison_medium(2).png"
)

plt.savefig(
    visual_path,
    dpi=300,
    bbox_inches="tight"
)


print("\n")
print("=" * 60)
print("VISUAL COMPARISON")
print("=" * 60)

print(f"Saved to:")
print(visual_path)

print("=" * 60)


# Show image
plt.show()


# ============================================================
# IMAGE PATHS
# ============================================================

print("\n")
print("=" * 60)
print("IMAGES USED")
print("=" * 60)

print(f"Clear   : {clear_path}")
print(f"Hazy    : {hazy_path}")
print(f"Dehazed : {dehazed_path}")
print("=" * 60)