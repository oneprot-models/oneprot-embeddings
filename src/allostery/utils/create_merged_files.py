"""
For each subfolder of 'embeddings':

POCKET variant (if ASD_pockets100 AND merged_pocket both exist):
  1. Copy merged_pocket → ASD_merged_pocket
  2. Rename .pt files inside each split folder to:
       ASD_merged_pocket_{split}_embeddings_labels.pt
  3. For each split, append all embeddings from:
       ASD_pockets100/{split}/ASD_pockets100_{split}_embeddings_labels.pt
     into the merged file's 'embeddings' key,
     with corresponding 'labels_fitness' values set to 1.

SEQUENCE variant (if ASD_pockets_sequence100 AND merged_pocket_sequence both exist):
  Same as above but with:
  - source:      merged_pocket_sequence
  - destination: ASD_merged_pocket_sequence
  - appended from: ASD_pockets_sequence100/{split}/ASD_pockets_sequence100_{split}_embeddings_labels.pt
  - new files named: ASD_merged_pocket_sequence_{split}_embeddings_labels.pt
"""

import shutil
import torch
from pathlib import Path


SPLITS = ["train", "test", "valid"]


def append_embeddings(merged_pt_path: Path, pockets100_pt_path: Path):
    """
    Load the merged .pt file and append all embeddings from pockets100.
    The appended embeddings get labels_fitness == 1.
    All other keys from pockets100 are also concatenated where possible.
    """
    print(f"    Loading merged:    {merged_pt_path.name}")
    merged = torch.load(merged_pt_path, map_location="cpu")

    print(f"    Loading pockets100: {pockets100_pt_path.name}")
    extra = torch.load(pockets100_pt_path, map_location="cpu")

    if not isinstance(merged, dict) or not isinstance(extra, dict):
        print("    WARNING: one of the files is not a dict, skipping append.")
        return

    print(f"    Keys in merged file: {list(merged.keys())}")
    n_extra = None

    # Append embeddings
    if "embeddings" not in extra:
        print("    WARNING: 'embeddings' key not found in pockets100 file, skipping.")
        return

    extra_embeddings = extra["embeddings"]
    n_extra = extra_embeddings.shape[0]
    print(f"    Appending {n_extra} embeddings from pockets100 ...")

    merged["embeddings"] = torch.cat([merged["embeddings"], extra_embeddings], dim=0)

    # Set labels_fitness = 1 for all appended rows
    if "labels_fitness" in merged:
        new_labels_fitness = torch.ones(n_extra, dtype=merged["labels_fitness"].dtype)
        merged["labels_fitness"] = torch.cat([merged["labels_fitness"], new_labels_fitness], dim=0)
    else:
        print("    WARNING: 'labels_fitness' key not found in merged file — creating it for appended rows only.")
        merged["labels_fitness"] = torch.ones(n_extra, dtype=torch.long)

    # For any other tensor keys in extra, concatenate them too (except embeddings already done)
    for key in extra:
        if key == "embeddings":
            continue
        if key not in merged:
            print(f"    Skipping key '{key}' (not in merged file).")
            continue
        if key == "labels_fitness":
            continue  # already handled above
        try:
            merged[key] = torch.cat([merged[key], extra[key]], dim=0)
            print(f"    Concatenated key: '{key}'")
        except Exception as e:
            print(f"    Could not concatenate key '{key}': {e}")

    torch.save(merged, merged_pt_path)
    print(f"    Saved: {merged_pt_path.name}")


def setup_merged_folder(
    folder: Path,
    src_name: str,
    dst_name: str,
    pockets100_name: str,
    merged_prefix: str,
    pockets100_prefix: str,
):
    src = folder / src_name
    dst = folder / dst_name
    pockets100 = folder / pockets100_name

    if not src.exists():
        print(f"  No '{src_name}' found, skipping.")
        return
    if not pockets100.exists():
        print(f"  No '{pockets100_name}' found, skipping.")
        return

    # Copy src → dst
    if dst.exists():
        print(f"  Removing existing '{dst_name}' ...")
        shutil.rmtree(dst)

    print(f"  Copying '{src_name}' → '{dst_name}' ...")
    shutil.copytree(src, dst)

    # Rename .pt files and append embeddings for each split
    for split in SPLITS:
        split_dir = dst / split
        if not split_dir.exists():
            print(f"  Split dir not found: {split_dir}, skipping.")
            continue

        # Find and rename the copied .pt file (it still has the old name from src)
        old_pt = None
        for f in split_dir.glob("*.pt"):
            old_pt = f
            break  # expect one .pt file per split dir

        new_pt_name = f"{merged_prefix}_{split}_embeddings_labels.pt"
        new_pt = split_dir / new_pt_name

        if old_pt is None:
            print(f"  No .pt file found in {split_dir}, skipping.")
            continue

        if old_pt.name != new_pt_name:
            print(f"  Renaming: {old_pt.name} → {new_pt_name}")
            old_pt.rename(new_pt)

        # Locate the pockets100 .pt file to append from
        pockets100_pt_name = f"{pockets100_prefix}_{split}_embeddings_labels.pt"
        pockets100_pt = pockets100 / split / pockets100_pt_name

        if not pockets100_pt.exists():
            print(f"  WARNING: pockets100 file not found: {pockets100_pt}, skipping append.")
            continue

        print(f"  Processing split '{split}':")
        append_embeddings(new_pt, pockets100_pt)


def main(embeddings_root: str):
    embeddings_root = Path(embeddings_root)
    if not embeddings_root.exists():
        raise FileNotFoundError(f"Not found: {embeddings_root}")

    subfolders = sorted([d for d in embeddings_root.iterdir() if d.is_dir()])
    print(f"Found {len(subfolders)} subfolders in '{embeddings_root}'\n")

    for folder in subfolders:
        print(f"{'='*60}")
        print(f"Folder: {folder.name}")

        # Pocket variant
        setup_merged_folder(
            folder=folder,
            src_name="merged_pocket",
            dst_name="ASD_merged_pocket",
            pockets100_name="ASD_pockets100",
            merged_prefix="ASD_merged_pocket",
            pockets100_prefix="ASD_pockets100",
        )

        # Sequence variant
        setup_merged_folder(
            folder=folder,
            src_name="merged_pocket_sequence",
            dst_name="ASD_merged_pocket_sequence",
            pockets100_name="ASD_pocket_sequence100",
            merged_prefix="ASD_merged_pocket_sequence",
            pockets100_prefix="ASD_pocket_sequence100",
        )

    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    import sys
    embeddings_path = sys.argv[1] if len(sys.argv) > 1 else "embeddings"
    main(embeddings_path)