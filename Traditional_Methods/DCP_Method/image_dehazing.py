import os
import cv2
import numpy as np
from scipy.ndimage import minimum_filter
import matplotlib.pyplot as plt


class ImageDehazer:
    """
    Image Dehazing using Dark Channel Prior (DCP)
    """

    def __init__(self, omega=0.80, t_min=0.25):
        self.omega = omega
        self.t_min = t_min
        self.A = None
        self.transmission = None

    def load_image(self, image_path):
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def rgb_separation(self, image):
        r_channel = image[:, :, 0]
        g_channel = image[:, :, 1]
        b_channel = image[:, :, 2]

        return r_channel, g_channel, b_channel

    def compute_dark_channel(self, image, window_size=15):

        if len(image.shape) == 3:
            dark_channel = np.min(image, axis=2)
        else:
            dark_channel = image

        dark_channel = minimum_filter(
            dark_channel,
            size=window_size
        )

        return dark_channel

    def estimate_atmospheric_light(
        self,
        image,
        dark_channel,
        top_percent=0.001
    ):

        h, w = dark_channel.shape

        num_pixels = max(
            1,
            int(h * w * top_percent)
        )

        flat_dark = dark_channel.flatten()

        flat_indices = np.argsort(
            flat_dark
        )[::-1][:num_pixels]

        rgb_values = image.reshape(-1, 3)[flat_indices]

        # Select the brightest candidate pixel
        brightness = np.sum(rgb_values, axis=1)

        best_index = np.argmax(brightness)

        A = rgb_values[best_index]

        return A.astype(np.float32)

    def estimate_transmission(
        self,
        image,
        A,
        window_size=15
    ):

        A = np.maximum(A, 0.1)

        normalized = (
            image.astype(np.float32) / A
        )

        dark_channel_normalized = (
            self.compute_dark_channel(
                normalized,
                window_size
            )
        )

        transmission = (
            1
            - self.omega
            * dark_channel_normalized
        )

        transmission = np.maximum(
            transmission,
            self.t_min
        )

        return transmission

    def refine_transmission(
        self,
        image,
        transmission,
        window_size=7
    ):

        # OpenCV medianBlur requires an odd kernel size
        transmission_refined = cv2.medianBlur(
            (transmission * 255).astype(np.uint8),
            window_size
        )

        transmission_refined = (
            transmission_refined.astype(np.float32)
            / 255.0
        )

        return transmission_refined

    def recover_scene_radiance(
        self,
        image,
        A,
        transmission
    ):

        transmission = np.maximum(
            transmission,
            self.t_min
        )

        if (
            len(image.shape) == 3
            and transmission.ndim == 2
        ):
            transmission = np.expand_dims(
                transmission,
                axis=2
            )

        radiance = (
            image.astype(np.float32) - A
        ) / transmission + A

        radiance = np.clip(
            radiance,
            0,
            255
        )

        return radiance.astype(np.uint8)

    def dehaze(self, image_path, refine=True):

        # 1. Load image
        image = self.load_image(image_path)

        print(f"Image shape: {image.shape}")

        # 2. RGB separation
        r, g, b = self.rgb_separation(image)

        print("RGB channels separated")

        # 3. Dark Channel Prior
        dark_channel = self.compute_dark_channel(
            image
        )

        print("Dark channel computed")

        # 4. Atmospheric Light
        self.A = self.estimate_atmospheric_light(
            image,
            dark_channel
        )

        print(
            f"Atmospheric light (A): {self.A}"
        )

        # 5. Transmission
        self.transmission = (
            self.estimate_transmission(
                image,
                self.A
            )
        )

        print("Transmission map estimated")

        # 6. Refine transmission
        if refine:

            self.transmission = (
                self.refine_transmission(
                    image,
                    self.transmission
                )
            )

            print(
                "Transmission map refined"
            )

        # 7. Recover scene radiance
        dehazed = (
            self.recover_scene_radiance(
                image,
                self.A,
                self.transmission
            )
        )

        print(
            "Scene radiance recovered"
        )

        return (
            image,
            dehazed,
            dark_channel,
            self.transmission
        )

    def visualize_results(
        self,
        original,
        dehazed,
        dark_channel,
        transmission,
        save_path
    ):

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(15, 10)
        )

        # Hazy image
        axes[0, 0].imshow(original)
        axes[0, 0].set_title(
            "Hazy Image (Input)"
        )
        axes[0, 0].axis("off")

        # Dark channel
        axes[0, 1].imshow(
            dark_channel,
            cmap="gray"
        )
        axes[0, 1].set_title(
            "Dark Channel Prior"
        )
        axes[0, 1].axis("off")

        # Transmission
        axes[0, 2].imshow(
            transmission,
            cmap="gray"
        )
        axes[0, 2].set_title(
            "Transmission Map t(x,y)"
        )
        axes[0, 2].axis("off")

        # Dehazed image
        axes[1, 0].imshow(dehazed)
        axes[1, 0].set_title(
            "Dehazed Image (Output)"
        )
        axes[1, 0].axis("off")

        # Before / after
        comparison = np.hstack(
            [original, dehazed]
        )

        axes[1, 1].imshow(comparison)
        axes[1, 1].set_title(
            "Before (left) vs After (right)"
        )
        axes[1, 1].axis("off")

        # Atmospheric light
        axes[1, 2].text(
            0.5,
            0.5,
            f"Atmospheric Light (A):\n"
            f"R: {self.A[0]:.2f}\n"
            f"G: {self.A[1]:.2f}\n"
            f"B: {self.A[2]:.2f}",
            ha="center",
            va="center",
            fontsize=12,
            family="monospace"
        )

        axes[1, 2].axis("off")

        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

        print(
            f"Visualization saved to:\n{save_path}"
        )

        plt.show()


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DARK CHANNEL PRIOR IMAGE DEHAZING")
    print("=" * 60)

    # ------------------------------------------------------
    # PATHS
    # ------------------------------------------------------

    haze_image_path = (
      r"C:\Users\patna\OneDrive\Desktop\IP_1\input_images\image1\low_haze.jpg"
    )

    results_folder = (
       r"C:\Users\patna\OneDrive\Desktop\IP_1\DCP_Results\image1\low_haze"
    )

    os.makedirs(
        results_folder,
        exist_ok=True
    )

    dehazed_output_path = os.path.join(
        results_folder,
        "dehazed_output.jpg"
    )

    visualization_path = os.path.join(
        results_folder,
        "dehazing_results.png"
    )

    # ------------------------------------------------------
    # INITIALIZE
    # ------------------------------------------------------

    dehazer = ImageDehazer(
        omega=0.80,
        t_min=0.25
    )

    try:

        # --------------------------------------------------
        # RUN DCP
        # --------------------------------------------------

        original, dehazed, dark_channel, transmission = (
            dehazer.dehaze(
                haze_image_path,
                refine=True
            )
        )

        # --------------------------------------------------
        # SAVE DEHAZED IMAGE
        # --------------------------------------------------

        dehazed_bgr = cv2.cvtColor(
            dehazed,
            cv2.COLOR_RGB2BGR
        )

        cv2.imwrite(
            dehazed_output_path,
            dehazed_bgr
        )

        print()
        print("=" * 60)
        print("DEHAZING COMPLETED")
        print("=" * 60)

        print(
            f"Dehazed image saved to:\n"
            f"{dehazed_output_path}"
        )

        # --------------------------------------------------
        # SAVE VISUALIZATION
        # --------------------------------------------------

        dehazer.visualize_results(
            original,
            dehazed,
            dark_channel,
            transmission,
            visualization_path
        )

    except FileNotFoundError as e:

        print()
        print("ERROR:")
        print(e)