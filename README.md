# Zircon Collage Processor with SAM

This project provides a command-line script that processes a collage image of zircons using Meta's Segment Anything Model (SAM).

It automatically detects whether a CUDA GPU is available:
- If a GPU is available, it runs on `cuda`.
- If no GPU is available, it falls back to `cpu`.

## What the script produces

Given:
- one input collage image, and
- one output directory,

the script writes:
1. **cropped zircon cutouts** (`zircon_001.png`, `zircon_002.png`, ... ) directly inside the output directory, and
2. a **`masks/` subfolder** with one binary mask per zircon (`zircon_001_mask.png`, ...).

## Requirements

- Python 3.9+
- `torch`
- `opencv-python`
- `numpy`
- `segment-anything`

Install dependencies:

```bash
pip install torch opencv-python numpy segment-anything
```

## Usage

```bash
python process_zircon_collage.py \
  /path/to/collage.png \
  /path/to/output_dir \
  --checkpoint /path/to/sam_vit_h_4b8939.pth \
  --model-type vit_h
```

### Optional arguments

- `--min-area` (default: `300`): filters out very small masks.
- `--points-per-side` (default: `32`): SAM sampling density; higher values can improve detail but increase runtime.

## Example output structure

```text
output_dir/
  zircon_001.png
  zircon_002.png
  ...
  masks/
    zircon_001_mask.png
    zircon_002_mask.png
    ...
```

## Notes

- SAM checkpoints can be downloaded from the official Segment Anything release resources.
- Make sure `--model-type` matches the checkpoint you provide.
- Large images or large SAM models may consume significant memory, especially on CPU.
