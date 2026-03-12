"""
For each folder in the 'embeddings' directory:
  - If ASD_pockets subfolder exists → copy to ASD_pockets_binary
  - If ASD_pockets_sequence subfolder exists → copy to ASD_pockets_sequence_binary

Then in each binary folder, for splits (test, train, valid):
  - Load {split}/ASD_pockets[_sequence]_{split}_embeddings_labels.pt
  - Change all labels == 2 → 1
  - Reverse all labels: 0 → 1, 1 → 0
  - Save back in place
"""

import os
import shutil
import torch
from pathlib import Path


def process_labels(tensor: torch.Tensor) -> torch.Tensor:
    """
    Step 1: Replace label value 2 with 1
    Step 2: Reverse labels (0 → 1, 1 → 0)
    """
    # Step 1: 2 → 1
    tensor = tensor.clone()
    tensor[tensor == 2] = 1

    # Step 2: Reverse (0 ↔ 1)
    tensor = 1 - tensor

    return tensor


def process_pt_file(pt_path: Path, labels_key: str = "labels_fitness"):
    """Load a .pt file, transform the labels, and save it back."""
    print(f"  Processing: {pt_path}")
    data = torch.load(pt_path, map_location="cpu")

    if isinstance(data, dict):
        if labels_key not in data:
            print(f"    WARNING: key '{labels_key}' not found. Keys: {list(data.keys())}")
            return
        original = data[labels_key]
        unique_before = original.unique().tolist()
        data[labels_key] = process_labels(data[labels_key])
        unique_after = data[labels_key].unique().tolist()
        print(f"    Labels before: {unique_before} → after: {unique_after}")
    else:
        # If the file is a plain tensor
        print(f"    WARNING: expected a dict but got {type(data)}. Skipping.")
        return

    torch.save(data, pt_path)
    print(f"    Saved.")


def process_binary_folder(binary_folder: Path, prefix: str):
    """
    Within a binary folder, process label files for each split.
    prefix is either 'ASD_pockets' or 'ASD_pockets_sequence'
    The target filename uses '{prefix}_binary_{split}_embeddings_labels.pt'.
    If the file was already copied with the old name (without _binary), rename it first.
    """
    binary_prefix = f"{prefix}_binary"
    splits = ["test", "train", "valid"]
    for split in splits:
        old_filename = f"{prefix}_{split}_embeddings_labels.pt"
        new_filename = f"{binary_prefix}_{split}_embeddings_labels.pt"
        old_path = binary_folder / split / old_filename
        new_path = binary_folder / split / new_filename

        # Rename old file to new name if it exists under the old name
        if old_path.exists():
            print(f"  Renaming: {old_filename} → {new_filename}")
            old_path.rename(new_path)

        if new_path.exists():
            process_pt_file(new_path)
        else:
            print(f"  SKIP (not found): {new_path}")


def main(embeddings_root: str):
    embeddings_root = Path(embeddings_root)
    if not embeddings_root.exists():
        raise FileNotFoundError(f"Embeddings root not found: {embeddings_root}")

    subfolders = [d for d in sorted(embeddings_root.iterdir()) if d.is_dir()]
    print(f"Found {len(subfolders)} folders in '{embeddings_root}':\n")

    for folder in subfolders:
        print(f"{'='*60}")
        print(f"Folder: {folder.name}")

        for prefix in ["ASD_pockets", "ASD_pockets_sequence"]:
            src = folder / prefix
            dst = folder / f"{prefix}_binary"

            if not src.exists():
                print(f"  No '{prefix}' subfolder found, skipping.")
                continue

            # Copy src → dst (overwrite if exists)
            if dst.exists():
                print(f"  Removing existing '{dst.name}' ...")
                shutil.rmtree(dst)

            print(f"  Copying '{prefix}' → '{prefix}_binary' ...")
            shutil.copytree(src, dst)
            print(f"  Copy done. Now transforming labels ...")

            process_binary_folder(dst, prefix)

    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Default: look for 'embeddings' folder in current directory
        embeddings_path = "embeddings"
    else:
        embeddings_path = sys.argv[1]

    main(embeddings_path)