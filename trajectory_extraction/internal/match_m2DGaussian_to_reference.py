#!/usr/bin/env python3
"""
match_m2DGaussian_to_reference.py

For each reference track ({G|P|R}_loci{N}_traj_rela2wholeimg.csv), find the
best-matching m2DGaussian candidate track of the same channel by computing the
average Euclidean distance across all shared frame numbers.  Matching is
one-to-one per channel (greedy by lowest average distance).  The matched
candidate is saved as:

    {G|P|R}_loci{N}_traj_m2DGaussian_cleaned.csv

in the same directory as the reference tracks.

The MATLAB candidate tracks are expected at:
    <ref_dir>/matlab_result/matlab_trajectory/{G|P|R}_m2DGaussian_traj*.csv

Usage:
    python3 match_m2DGaussian_to_reference.py <ref_dir>

Example:
    python3 match_m2DGaussian_to_reference.py \
        "/Users/chenxinyi/Desktop/LIVEFISH analysis/published/published_2/FOV5_analyzed_1_copy10_count/try_analysis"
"""

import csv
import math
import sys
from pathlib import Path

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PARAMETERS                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Minimum number of shared frames required for a candidate to be considered.
MIN_OVERLAP_FRAMES = 5

# Maximum allowed average distance (nm) between a reference and matched candidate.
# Pairs exceeding this are rejected even if they are the closest available match.
MAX_AVG_DIST_NM = 2000

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  IMPLEMENTATION                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_track(path: Path) -> dict:
    """Load a trajectory CSV → {frame_int: (x_nm, y_nm)}."""
    track = {}
    with open(str(path), newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row['frame'])
            track[frame] = (float(row['x_nm']), float(row['y_nm']))
    return track


def avg_distance(ref: dict, cand: dict) -> tuple:
    """
    Compute average Euclidean distance over shared frames.
    Returns (avg_dist, n_shared).  Returns (inf, 0) if shared frames < MIN_OVERLAP_FRAMES.
    """
    shared = set(ref.keys()) & set(cand.keys())
    if len(shared) < MIN_OVERLAP_FRAMES:
        return math.inf, len(shared)
    total = sum(
        math.hypot(ref[t][0] - cand[t][0], ref[t][1] - cand[t][1])
        for t in shared
    )
    return total / len(shared), len(shared)


def extract_loci_number(path: Path) -> str:
    """Extract the loci number string from a reference filename like G_loci2_traj_..."""
    # filename: G_loci2_traj_rela2wholeimg.csv  →  '2'
    name = path.stem  # G_loci2_traj_rela2wholeimg
    parts = name.split('_')
    for part in parts:
        if part.startswith('loci'):
            return part[len('loci'):]
    return name


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 match_m2DGaussian_to_reference.py <ref_dir>")
        sys.exit(1)

    ref_dir    = Path(sys.argv[1]).resolve()
    matlab_dir = ref_dir / "matlab_result" / "matlab_trajectory"

    if not ref_dir.is_dir():
        print(f"ERROR: ref_dir not found: {ref_dir}")
        sys.exit(1)
    if not matlab_dir.is_dir():
        print(f"ERROR: MATLAB trajectory dir not found: {matlab_dir}")
        sys.exit(1)

    print(f"Ref dir   : {ref_dir}")
    print(f"MATLAB dir: {matlab_dir}")

    total_refs = total_matched = 0

    for channel in ('G', 'P', 'R'):
        print(f"\n{'═'*70}")
        print(f"Channel: {channel}")
        print(f"{'═'*70}")

        # ── Load reference tracks for this channel ────────────────────────────
        ref_paths = sorted(ref_dir.glob(f"{channel}_loci*_traj_rela2wholeimg.csv"))
        if not ref_paths:
            print(f"  No reference {channel} tracks found — skipping.")
            continue
        print(f"  Reference tracks ({len(ref_paths)}):")
        for p in ref_paths:
            print(f"    {p.name}")

        # ── Load candidate tracks for this channel ────────────────────────────
        cand_paths = sorted(matlab_dir.glob(f"{channel}_m2DGaussian_traj*.csv"))
        if not cand_paths:
            print(f"  No m2DGaussian {channel} tracks found — skipping.")
            continue
        print(f"  Candidate tracks ({len(cand_paths)}):")
        for p in cand_paths:
            print(f"    {p.name}")

        refs  = {p: load_track(p) for p in ref_paths}
        cands = {p: load_track(p) for p in cand_paths}

        # ── Score all pairs ───────────────────────────────────────────────────
        print(f"\n  Scoring all pairs (MIN_OVERLAP_FRAMES={MIN_OVERLAP_FRAMES})...")
        scored = []
        for rp, rtrack in refs.items():
            for cp, ctrack in cands.items():
                dist, n_shared = avg_distance(rtrack, ctrack)
                if dist < math.inf:
                    scored.append((dist, n_shared, rp, cp))

        scored.sort(key=lambda x: x[0])

        # ── Greedy one-to-one matching ────────────────────────────────────────
        matched_refs  = set()
        matched_cands = set()
        assignments   = {}  # ref_path → (cand_path, avg_dist, n_shared)

        for dist, n_shared, rp, cp in scored:
            if rp in matched_refs or cp in matched_cands:
                continue
            matched_refs.add(rp)
            matched_cands.add(cp)
            assignments[rp] = (cp, dist, n_shared)

        # ── Write output and print summary ────────────────────────────────────
        print(f"\n  {'─'*66}")
        print(f"  {'Locus':<10}  {'Matched candidate':<30}  {'Shared':>6}  {'Avg dist (nm)':>14}")
        print(f"  {'─'*66}")

        for rp in ref_paths:
            loci_num = extract_loci_number(rp)
            if rp not in assignments:
                print(f"  loci{loci_num:<6}  {'NO MATCH (insufficient overlap)':<30}")
                continue

            cp, dist, n_shared = assignments[rp]
            if dist > MAX_AVG_DIST_NM:
                print(f"  loci{loci_num:<6}  {cp.name:<30}  {n_shared:>6}  {dist:>14.2f}  ✗ REJECTED (>{MAX_AVG_DIST_NM} nm)")
                continue

            # For red channel: reject if the MATLAB trajectory has no points
            # in the first 3 frames (1-indexed frames 1, 2, 3).
            ctrack = cands[cp]
            if channel == 'R' and not any(f <= 3 for f in ctrack):
                print(f"  loci{loci_num:<6}  {cp.name:<30}  {n_shared:>6}  {dist:>14.2f}"
                      f"  ✗ REJECTED (first 3 frames missing in MATLAB track)")
                continue

            out_path = ref_dir / f"{channel}_loci{loci_num}_traj_m2DGaussian_cleaned.csv"

            with open(str(out_path), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame', 'x_nm', 'y_nm'])
                for frame in sorted(ctrack.keys()):
                    x_nm, y_nm = ctrack[frame]
                    writer.writerow([frame, f'{x_nm:.2f}', f'{y_nm:.2f}'])

            print(f"  loci{loci_num:<6}  {cp.name:<30}  {n_shared:>6}  {dist:>14.2f}")
            print(f"            → saved: {out_path.name}")

        print(f"  {'─'*66}")
        print(f"  {len(assignments)}/{len(ref_paths)} {channel} loci matched.")
        total_refs    += len(ref_paths)
        total_matched += len(assignments)

    print(f"\n{'═'*70}")
    print(f"Total: {total_matched}/{total_refs} loci matched across all channels.")
    print("Done.")


if __name__ == '__main__':
    main()
