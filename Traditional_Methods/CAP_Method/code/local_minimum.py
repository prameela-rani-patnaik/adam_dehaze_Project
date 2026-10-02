import numpy as np

def local_minimum(image):
    H, W = image.shape

    patch_size = 15
    pad_size = patch_size // 2

    local_min = np.zeros((H, W), dtype=np.float32)

    imJ = np.pad(
        image,
        ((pad_size, pad_size),
         (pad_size, pad_size)),
        mode='constant',
        constant_values=np.inf
    )

    for j in range(H):
        for i in range(W):
            patch = imJ[
                j:j + patch_size,
                i:i + patch_size
            ]

            local_min[j, i] = np.min(patch)

    return local_min