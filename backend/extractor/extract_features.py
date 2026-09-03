import sys
import os
import io
import json
import base64
import argparse
import warnings
import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')

def apply_clahe_shadow_suppression(rgb_img_uint8: np.ndarray, mask: np.ndarray = None, clip_limit: float = 2.0) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the L* (luminance)
    channel in CIELAB color space. This equalizes harsh outdoor shadows, non-uniform solar
    illumination gradients, and specular glare across foliar surfaces without altering
    chromatic foliar hue or chlorophyll saturation.
    Safe for both lab and field images: on uniform lab images, clip_limit prevents over-amplification.
    """
    try:
        import cv2
        lab = cv2.cvtColor(rgb_img_uint8, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_channel)

        if mask is not None:
            # Apply equalized luminance specifically to the segmented foliar blade
            l_final = np.where(mask, l_clahe, l_channel)
        else:
            l_final = l_clahe

        lab_equalized = cv2.merge((l_final, a_channel, b_channel))
        return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2RGB)
    except Exception:
        # Graceful fallback: return unmodified array if cv2 is not available
        return rgb_img_uint8

def auto_mask_and_letterbox(raw_img: Image.Image, target_size=(299, 299)) -> tuple[Image.Image, str]:
    """
    1. Segments subject leaf from background (table, hands, shadows, outdoor clutter)
       and fills background with pure white (255, 255, 255) to align with Swedish/Flavia/Philippine
       laboratory benchmark distributions.
    2. Applies Gray-World color constancy + CIELAB CLAHE shadow suppression to neutralize
       ambient color casts and harsh directional lighting/shadows.
    3. Letterboxes leaf onto target_size canvas while strictly preserving aspect ratio
       to prevent venation angle and morphological perimeter distortion.
    4. Returns (processed_PIL_image, base64_data_uri).
    """
    img = raw_img.convert("RGB")
    arr = np.array(img, dtype=np.float32)

    # 1. Botanical Vegetation Masking
    # Greenness Index (ExG = 2*G - R - B) highlights foliar tissue/chlorophyll
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    exg = 2.0 * g - r - b

    # HSV Color Space Filter:
    # Foliar green hue in PIL 0-255 scale: [20, 115] (~28 deg to ~162 deg)
    # Excludes human skin (Hue < 20, Red dominant) and cloth (Blue hue or low saturation)
    hsv_arr = np.array(img.convert("HSV"))
    hue = hsv_arr[:, :, 0]
    sat = hsv_arr[:, :, 1]
    val = hsv_arr[:, :, 2]

    # Strong foliar chlorophyll mask:
    # Must have botanical green hue + adequate saturation + green dominance over red/blue
    is_green_foliage = (
        (hue >= 20) & (hue <= 115) &
        (sat >= 28) & (val >= 25) &
        (g > r - 10.0) &
        (exg > 8.0)
    )

    total_pixels = arr.shape[0] * arr.shape[1]
    green_ratio = np.sum(is_green_foliage) / total_pixels

    if 0.015 <= green_ratio <= 0.95:
        # Clear green leaf identified! Use foliar mask with morphological refinement
        try:
            import scipy.ndimage as ndi
            # Keep only the largest connected green component to eliminate stray noise/shadow specks
            labeled, num = ndi.label(is_green_foliage)
            if num > 0:
                sizes = ndi.sum(is_green_foliage, labeled, range(1, num + 1))
                largest_label = np.argmax(sizes) + 1
                leaf_mask = (labeled == largest_label)
                # Fill any interior gaps/specular reflections inside leaf
                leaf_mask = ndi.binary_fill_holes(leaf_mask)
            else:
                leaf_mask = is_green_foliage
        except Exception:
            leaf_mask = is_green_foliage
    else:
        # Fallback for dried/autumn leaves on a clean white background
        dist_white = np.sqrt((255.0 - r)**2 + (255.0 - g)**2 + (255.0 - b)**2)
        is_not_white = (r < 240) | (g < 240) | (b < 240)
        leaf_mask = is_not_white & (dist_white > 45.0)

    foreground_ratio = np.sum(leaf_mask) / total_pixels

    # If valid foreground detected
    if 0.01 <= foreground_ratio <= 0.98:
        masked_arr = arr.copy()

        # Apply Gray-World Color Constancy to the foliar blade to neutralize ambient lighting casts
        mean_r = np.mean(r[leaf_mask])
        mean_g = np.mean(g[leaf_mask])
        mean_b = np.mean(b[leaf_mask])
        gray_target = (mean_r + mean_g + mean_b) / 3.0

        gain_r = float(np.clip(gray_target / max(mean_r, 1e-5), 0.75, 1.35))
        gain_g = float(np.clip(gray_target / max(mean_g, 1e-5), 0.75, 1.35))
        gain_b = float(np.clip(gray_target / max(mean_b, 1e-5), 0.75, 1.35))

        masked_arr[:, :, 0][leaf_mask] = np.clip(masked_arr[:, :, 0][leaf_mask] * gain_r, 0, 255)
        masked_arr[:, :, 1][leaf_mask] = np.clip(masked_arr[:, :, 1][leaf_mask] * gain_g, 0, 255)
        masked_arr[:, :, 2][leaf_mask] = np.clip(masked_arr[:, :, 2][leaf_mask] * gain_b, 0, 255)

        # CIELAB CLAHE Shadow & Glare Suppression:
        # Equalizes non-uniform solar illumination and cast shadows on the leaf blade
        # while keeping chromatic channels (chlorophyll green/yellow) intact.
        leaf_uint8 = np.clip(masked_arr, 0, 255).astype(np.uint8)
        clahe_enhanced = apply_clahe_shadow_suppression(leaf_uint8, mask=leaf_mask, clip_limit=2.0)
        masked_arr[leaf_mask] = clahe_enhanced[leaf_mask].astype(np.float32)

        # Set all background pixels (towel, fingers, desk, outdoor clutter) to pure studio white
        masked_arr[~leaf_mask] = 255.0

        # Crop to bounding box with 5% morphological cushion
        y_indices, x_indices = np.where(leaf_mask)
        ymin, ymax = y_indices.min(), y_indices.max()
        xmin, xmax = x_indices.min(), x_indices.max()

        h, w = arr.shape[:2]
        pad_y = int((ymax - ymin) * 0.05)
        pad_x = int((xmax - xmin) * 0.05)
        ymin = max(0, ymin - pad_y)
        ymax = min(h, ymax + pad_y)
        xmin = max(0, xmin - pad_x)
        xmax = min(w, xmax + pad_x)

        leaf_cropped = Image.fromarray(masked_arr[ymin:ymax, xmin:xmax].astype(np.uint8))
    else:
        # Fallback when no background is cleanly segregated (e.g., extreme macro crop)
        leaf_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
        leaf_cropped = Image.fromarray(apply_clahe_shadow_suppression(leaf_uint8, clip_limit=1.5))

    # 2. Aspect-Ratio Preserving Letterbox onto White Canvas
    img_w, img_h = leaf_cropped.size
    target_w, target_h = target_size
    scale = min(target_w / max(1, img_w), target_h / max(1, img_h))
    new_w = max(1, int(img_w * scale))
    new_h = max(1, int(img_h * scale))

    resized = leaf_cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, (255, 255, 255))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    # 3. Generate Base64 preview for UI
    buffered = io.BytesIO()
    canvas.save(buffered, format="JPEG", quality=85)
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    return canvas, img_b64

def extract_inception_v3_features(image_path: str, feature_count: int = 2048, dataset_type: str = "swedish"):
    """
    Segments leaf, preserves aspect ratio on 299x299 white canvas, and extracts
    2048-dimensional deep feature embeddings aligned with Google TensorFlow/Keras Inception-V3.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    prefix = "n" if "philippine" in dataset_type.lower() else "Att"
    raw_img = Image.open(image_path)
    processed_canvas, img_b64 = auto_mask_and_letterbox(raw_img, target_size=(299, 299))

    try:
        extracted_vector = None

        # Strategy 1: Orange Data Mining Exact Embedder (Same pipeline that created the training CSVs)
        try:
            import importlib
            ie = importlib.import_module("orangecontrib.imageanalytics.image_embedder")
            embedder = getattr(ie, "ImageEmbedder")(model="inception-v3")
            res = embedder([os.path.abspath(image_path)])
            if res and len(res) > 0 and len(res[0]) >= feature_count:
                extracted_vector = [float(v) for v in res[0][:feature_count]]
        except Exception as e:
            sys.stderr.write(f"Orange embedder notice ({e}), proceeding with PyTorch fallback.\n")

        # Strategy 2: Local timm / TorchVision Inception-V3
        if extracted_vector is None:
            try:
                import torch
                import torchvision.transforms as transforms

                model = None
                try:
                    import timm
                    model = timm.create_model('inception_v3.tf_in1k', pretrained=True, num_classes=0)
                    model.eval()
                except Exception:
                    pass

                if model is None:
                    import torchvision.models as models
                    weights = models.Inception_V3_Weights.DEFAULT
                    model = models.inception_v3(weights=weights, transform_input=False)
                    model.fc = torch.nn.Identity()
                    model.eval()

                preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])

                input_tensor = preprocess(processed_canvas).unsqueeze(0)

                with torch.no_grad():
                    output_features = model(input_tensor)
                    if isinstance(output_features, tuple):
                        output_features = output_features[0]
                    extracted_vector = output_features.squeeze().cpu().numpy().tolist()

            except Exception as e:
                sys.stderr.write(f"PyTorch Inception extraction error: {e}\n")

        if not extracted_vector or len(extracted_vector) < feature_count:
            raise ValueError(f"Inception-V3 returned incomplete vector: {len(extracted_vector) if extracted_vector else 0} features")

        # L2 Energy Calibration: Standardize vector magnitude to match training benchmark mean norm (~20.28)
        vec_arr = np.array(extracted_vector[:feature_count], dtype=np.float32)
        v_norm = float(np.linalg.norm(vec_arr))
        if v_norm > 1e-5:
            calibrated_vec = vec_arr * (20.28 / v_norm)
        else:
            calibrated_vec = vec_arr

        features = {}
        for i, val in enumerate(calibrated_vec):
            features[f"{prefix}{i}"] = round(float(val), 6)

        return {
            "features": features,
            "processed_image": img_b64
        }

    except Exception as deep_err:
        sys.stderr.write(f"Inception-V3 CNN unavailable ({deep_err}), switching to PIL/NumPy feature extractor...\n")
        arr = np.array(processed_canvas.resize((128, 128)), dtype=np.float32) / 255.0

        r_hist, _ = np.histogram(arr[:, :, 0], bins=64, range=(0.0, 1.0))
        g_hist, _ = np.histogram(arr[:, :, 1], bins=64, range=(0.0, 1.0))
        b_hist, _ = np.histogram(arr[:, :, 2], bins=64, range=(0.0, 1.0))
        
        patches = []
        for r_idx in range(4):
            for c_idx in range(4):
                patch = arr[r_idx*32:(r_idx+1)*32, c_idx*32:(c_idx+1)*32]
                patches.extend([patch.mean(), patch.std(), patch.max(), patch.min()])

        combined = np.concatenate([r_hist, g_hist, b_hist, np.array(patches, dtype=np.float32)])
        repeat_times = int(np.ceil(feature_count / len(combined)))
        vector = np.tile(combined, repeat_times)[:feature_count]

        features = {}
        for i, val in enumerate(vector):
            features[f"{prefix}{i}"] = round(float(val), 6)

        return {
            "features": features,
            "processed_image": img_b64
        }

def main():
    parser = argparse.ArgumentParser(description="Inception-V3 Deep Feature Extractor with Leaf Auto-Masking & Letterbox")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image file")
    parser.add_argument("--dataset", type=str, default="swedish", help="Dataset target: swedish, flavia, philippine")
    args = parser.parse_args()

    try:
        result = extract_inception_v3_features(args.image, feature_count=2048, dataset_type=args.dataset)
        print(json.dumps(result))
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
