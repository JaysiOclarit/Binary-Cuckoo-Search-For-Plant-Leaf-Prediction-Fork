import sys
import os
import json
import argparse
import warnings

warnings.filterwarnings('ignore')

def extract_inception_v3_features(image_path: str, feature_count: int = 2048, dataset_type: str = "swedish"):
    """
    Extracts 2048-dimensional deep feature embeddings using PyTorch Inception-V3 CNN.
    Maps extracted continuous feature vectors to Att0...Att2047 or n0...n2047.
    Raises an exception if image loading or feature extraction fails.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image

    # Load pretrained Inception-V3 CNN model backbone
    weights = models.Inception_V3_Weights.DEFAULT
    model = models.inception_v3(weights=weights, transform_input=False)
    model.fc = torch.nn.Identity() # Output 2048-dim bottleneck vector
    model.eval()

    # Keras/Orange Inception-V3 exact image preprocessing ([-1.0, 1.0] normalization)
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        output_features = model(input_tensor)
        if isinstance(output_features, tuple):
            output_features = output_features[0]
        extracted_vector = output_features.squeeze().cpu().numpy().tolist()

    if not extracted_vector or len(extracted_vector) < feature_count:
        raise ValueError(f"Inception-V3 feature extraction returned incomplete vector: {len(extracted_vector)} features")

    prefix = "n" if "philippine" in dataset_type.lower() else "Att"
    features = {}
    for i, val in enumerate(extracted_vector[:feature_count]):
        features[f"{prefix}{i}"] = round(float(val), 6)

    return features

def main():
    parser = argparse.ArgumentParser(description="Inception-V3 Deep Feature Extractor for Plant Leaf Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image file")
    parser.add_argument("--dataset", type=str, default="swedish", help="Dataset target: swedish, flavia, philippine")
    args = parser.parse_args()

    try:
        features = extract_inception_v3_features(args.image, feature_count=2048, dataset_type=args.dataset)
        print(json.dumps(features))
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
