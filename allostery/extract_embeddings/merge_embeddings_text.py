"""
merge_text_embeddings.py  —  run on CPU (no GPU or model loading needed)

Merges text-augmented embedding .pt files across ALL model subfolders under
EMBEDDINGS_ROOT that contain both Kinase and ASD text embedding folders:

  Kinase_pocket_text          + ASD_pockets_binary_text          -> merged_pocket_binary_text
  Kinase_combined_text        + ASD_pockets_sequence_binary_text  -> merged_pocket_sequence_binary_text

Label remapping to unified scheme (allosteric=1, non-allosteric=0):

  Kinase source:
    original 0 = competitive  -> 0 (non-allosteric, no change)
    original 1 = allosteric   -> 1 (allosteric, no change)

  ASD binary source (after process_labels: 2->1 then 0<->1):
    original allosteric(0)       -> flipped to 1 (allosteric)      -> 1 (no change)
    original competitive/non(1)  -> flipped to 0 (non-allosteric)  -> 0 (no change)

Both sources already use allosteric=1, non-allosteric=0 after the binary
transform, so the remap is identity. The explicit remap dict below makes
this auditable and easy to change if needed.

No model loading required — pure tensor manipulation.
"""

import os
import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDINGS_ROOT = 'embeddings'
SPLITS          = ['train', 'valid', 'test']

# (kinase_source_folder, asd_source_folder, output_folder)
MERGE_PAIRS = [
    ('Kinase_pocket_text',    'ASD_pockets_text',          'merged_pocket_text'),
    ('Kinase_combined_text',  'ASD_pockets_sequence_text',  'merged_pocket_sequence_text'),
    ('Kinase_pocket',         'ASD_pockets',                       'merged_pocket'),
    ('Kinase_combined',       'ASD_pockets_sequence',              'merged_pocket_sequence'),
]

# Explicit label remapping for each source.
# Key: original label value  ->  Value: unified label value.
# Both are already allosteric=1 / non-allosteric=0, so identity maps.
# Edit here if your sources use different conventions.
KINASE_LABEL_MAP = {0: 0, 1: 1}   # competitive->0, allosteric->1
ASD_LABEL_MAP    = {0: 0, 1: 1}   # non-allosteric->0, allosteric->1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def remap_labels(labels: torch.Tensor, label_map: dict) -> torch.Tensor:
    """Apply a {old -> new} integer label map to a 1-D label tensor."""
    out = labels.clone()
    for old_val, new_val in label_map.items():
        out[labels == old_val] = new_val
    return out


def merge_and_save(kinase_pt: str, asd_pt: str, output_pt: str,
                   label_str: str) -> bool:
    kinase = torch.load(kinase_pt, map_location='cpu')
    asd    = torch.load(asd_pt,    map_location='cpu')

    k_emb = kinase['embeddings']
    k_lbl = remap_labels(kinase['labels_fitness'], KINASE_LABEL_MAP)
    a_emb = asd['embeddings']
    a_lbl = remap_labels(asd['labels_fitness'], ASD_LABEL_MAP)

    if k_emb.shape[1] != a_emb.shape[1]:
        print(f"  ERROR [{label_str}]: embedding dims don't match — "
              f"Kinase={k_emb.shape[1]}, ASD={a_emb.shape[1]}. Skipping.")
        return False

    out_emb = torch.cat([k_emb, a_emb], dim=0)
    out_lbl = torch.cat([k_lbl, a_lbl], dim=0)

    os.makedirs(os.path.dirname(output_pt), exist_ok=True)
    tmp = output_pt + f'.{os.getpid()}.tmp'
    torch.save({'embeddings': out_emb, 'labels_fitness': out_lbl}, tmp)
    os.replace(tmp, output_pt)

    unique, counts = torch.unique(out_lbl, return_counts=True)
    dist = {int(k): int(c) for k, c in zip(unique, counts)}
    print(f"  Saved shape={out_emb.shape} -> {output_pt}")
    print(f"  Sources: Kinase={len(k_lbl)}, ASD={len(a_lbl)}  |  "
          f"Label distribution: {dist}")
    return True


def find_model_dirs(root: str) -> list[str]:
    """
    Return all subdirectories of root that contain at least one complete
    merge pair (both kinase source and ASD source present).
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"EMBEDDINGS_ROOT not found: {root}")

    eligible = []
    for name in sorted(os.listdir(root)):
        model_dir = os.path.join(root, name)
        if not os.path.isdir(model_dir):
            continue
        has_any_pair = any(
            os.path.isdir(os.path.join(model_dir, k_src)) and
            os.path.isdir(os.path.join(model_dir, a_src))
            for k_src, a_src, _ in MERGE_PAIRS
        )
        if has_any_pair:
            eligible.append(model_dir)
    return eligible


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    model_dirs = find_model_dirs(EMBEDDINGS_ROOT)
    print(f"Found {len(model_dirs)} eligible model subdirectorie(s):")
    for d in model_dirs:
        print(f"  {d}")

    total_written = 0
    total_skipped = 0

    #for model_dir in model_dirs:
    for model_dir in ['embeddings/oneprot_md_combined_gpcr_no_struct_graph_32900']:
        model_name = os.path.basename(model_dir)
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 70}")

        for k_folder, a_folder, out_folder in MERGE_PAIRS:
            k_base = os.path.join(model_dir, k_folder)
            a_base = os.path.join(model_dir, a_folder)

            print(f"\n  {k_folder}")
            print(f"  + {a_folder}")
            print(f"  -> {out_folder}")

            if not os.path.isdir(k_base):
                print(f"  Kinase source not found: {k_base}, skipping.")
                total_skipped += len(SPLITS)
                continue
            if not os.path.isdir(a_base):
                print(f"  ASD source not found: {a_base}, skipping.")
                total_skipped += len(SPLITS)
                continue

            for split in SPLITS:
                print(f"\n  [{split.upper()}]")

                k_pt  = os.path.join(k_base, split,
                                     f"{k_folder}_{split}_embeddings_labels.pt")
                a_pt  = os.path.join(a_base, split,
                                     f"{a_folder}_{split}_embeddings_labels.pt")
                out_pt = os.path.join(model_dir, out_folder, split,
                                      f"{out_folder}_{split}_embeddings_labels.pt")

                missing = [p for p in [k_pt, a_pt] if not os.path.exists(p)]
                if missing:
                    for p in missing:
                        print(f"    Not found: {p}")
                    total_skipped += 1
                    continue

                success = merge_and_save(
                    k_pt, a_pt, out_pt,
                    label_str=f"{model_name}/{out_folder}/{split}"
                )
                if success:
                    total_written += 1
                else:
                    total_skipped += 1

    print(f"\n{'=' * 70}")
    print(f"Done!  Written: {total_written}  Skipped/failed: {total_skipped}")
    print(f"{'=' * 70}")

    print("\nOutput files:")
    for model_dir in model_dirs:
        for _, _, out_folder in MERGE_PAIRS:
            for split in SPLITS:
                path = os.path.join(model_dir, out_folder, split,
                                    f"{out_folder}_{split}_embeddings_labels.pt")
                status = "✓" if os.path.exists(path) else "✗"
                print(f"  {status} {path}")