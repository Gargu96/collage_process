#!/usr/bin/env python3
"""Extract zircon cutouts and masks from a collage image using Segment Anything (SAM)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process a zircon collage image with SAM and export zircon cutouts plus "
            "individual binary masks."
        )
    )
    parser.add_argument(
        "input_image",
        type=Path,
        help="Path to the input collage image.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Output directory. Cutouts are written at this level and masks are written "
            "to output_dir/masks/."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the SAM checkpoint (.pth).",
    )
    parser.add_argument(
        "--model-type",
        default="vit_h",
        choices=("vit_h", "vit_l", "vit_b"),
        help="SAM model type that matches the checkpoint (default: vit_h).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=300,
        help="Discard masks smaller than this area in pixels (default: 300).",
    )
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=32,
        help="SAM automatic generator points_per_side value (default: 32).",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.input_image.is_file():
        raise FileNotFoundError(f"Input image was not found: {args.input_image}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"SAM checkpoint was not found: {args.checkpoint}")


def resolve_device() -> str:
    """Select GPU when available, otherwise use CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_image_rgb(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_mask_generator(
    model_type: str,
    checkpoint: Path,
    device: str,
    points_per_side: int,
) -> SamAutomaticMaskGenerator:
    model = sam_model_registry[model_type](checkpoint=str(checkpoint))
    model.to(device=device)
    return SamAutomaticMaskGenerator(model=model, points_per_side=points_per_side)


def save_outputs(
    image_rgb: np.ndarray,
    masks: list[dict],
    output_dir: Path,
    min_area: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    for idx, item in enumerate(masks, start=1):
        area = int(item.get("area", 0))
        if area < min_area:
            continue

        segmentation = item["segmentation"].astype(np.uint8)
        x, y, w, h = map(int, item["bbox"])

        patch = image_rgb[y : y + h, x : x + w]
        patch_mask = segmentation[y : y + h, x : x + w]

        cutout_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        cutout_rgba[..., :3] = patch
        cutout_rgba[..., 3] = patch_mask * 255

        stem = f"zircon_{idx:03d}"
        cutout_path = output_dir / f"{stem}.png"
        mask_path = masks_dir / f"{stem}_mask.png"

        cv2.imwrite(str(cutout_path), cv2.cvtColor(cutout_rgba, cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(str(mask_path), patch_mask * 255)
        kept += 1

    return kept


def main() -> None:
    args = parse_args()
    validate_inputs(args)

    device = resolve_device()
    print(f"Running SAM on device: {device}")

    image_rgb = load_image_rgb(args.input_image)
    generator = build_mask_generator(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        device=device,
        points_per_side=args.points_per_side,
    )

    print("Generating masks...")
    masks = generator.generate(image_rgb)
    print(f"Masks produced by SAM: {len(masks)}")

    kept = save_outputs(
        image_rgb=image_rgb,
        masks=masks,
        output_dir=args.output_dir,
        min_area=args.min_area,
    )

    print(f"Saved {kept} zircon cutouts to: {args.output_dir}")
    print(f"Saved corresponding masks to: {args.output_dir / 'masks'}")


if __name__ == "__main__":
    main()
