"""
Batch-process 25 .msh files x 15 case.sif files through ElmerGrid +
ElmerSolver, with live progress/ETA printed to the console.

RESUME SUPPORT (new):
  Every successful ElmerSolver run writes a ".completed" marker file
  inside its variant_XX folder. On a fresh run of this script, any
  variant that already has that marker is skipped entirely (no
  ElmerGrid, no ElmerSolver, nothing touched). Any variant folder
  that does NOT have the marker (whether it was never started, or
  was interrupted partway through) is (re)run from scratch --
  its folder is wiped and rebuilt cleanly first, so there's no
  risk of stale/partial files lingering around.

Folder structure produced:
  results/
    <sif_name_1>/
      variant_01/
        mesh_db/     <- ElmerGrid 14 2 output for that mesh
        results/     <- ElmerSolver VTU output (per SIF's Results Directory)
        case.sif     <- template, patched to point Mesh DB at "mesh_db"
        run_log.txt
        .completed   <- written only after a fully successful run
      variant_02/
        ...
    <sif_name_2>/
      variant_01/
      ...

Requires: ElmerGrid and ElmerSolver available on PATH.
"""

import re
import shutil
import subprocess
import time
import datetime
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════
# CONFIG — edit these paths to match your setup
# ══════════════════════════════════════════════════════════════════════════

MSH_INPUT_DIR = Path(
    r"D:\NeurIPS\surrogate_block\block_mesh_outputs"
)
SIF_INPUT_DIR = Path(
    r"D:\NeurIPS\surrogate_block\block_sif_cases_1033"
)
OUTPUT_ROOT = Path(
    r"D:\NeurIPS\surrogate_block\results_case_1033"
)

# Cache directory for one-time mesh conversions (see OPTIMIZATION note above)
MESH_CACHE_DIR = Path(
    r"D:\NeurIPS\surrogate_block\mesh_cache"
)
REUSE_MESH_CONVERSION = False  # ElmerGrid now reruns fresh every combination, no caching

MESH_DIR_NAME = "mesh_db"  # matches your requested folder name

ELMERGRID_EXE   = "ElmerGrid"
ELMERSOLVER_EXE = "ElmerSolver"

SOLVER_TIMEOUT_SECONDS = 4 * 3600

# How often (seconds) to print a step-progress/ETA line while a single
# ElmerSolver run is in progress. Doesn't affect the log file, which
# still captures every line.
PROGRESS_PRINT_INTERVAL_SECONDS = 15

COMPLETED_MARKER_NAME = ".completed"

# ══════════════════════════════════════════════════════════════════════════

# Matches Elmer's standard timestep progress line, e.g. "Time: 45/41420"
STEP_PATTERN = re.compile(r"\bTime:\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def format_duration(seconds):
    """Human-readable H:MM:SS from a seconds count. Handles None/negative gracefully."""
    if seconds is None or seconds < 0:
        return "unknown"
    return str(datetime.timedelta(seconds=int(seconds)))


def find_files(directory, pattern):
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found in {directory}")
    return files


def run_command_simple(cmd, cwd, log_file, timeout=None):
    """Run a subprocess quietly (used for ElmerGrid — fast, no need for live progress)."""
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        log.flush()
        try:
            result = subprocess.run(
                cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                timeout=timeout, text=True,
            )
        except subprocess.TimeoutExpired:
            log.write(f"\n!! TIMED OUT after {timeout}s\n")
            return False
        except FileNotFoundError as e:
            log.write(f"\n!! COMMAND NOT FOUND: {e}\n")
            return False

        if result.returncode != 0:
            log.write(f"\n!! Exited with return code {result.returncode}\n")
            return False
    return True


def run_solver_with_progress(cmd, cwd, log_file, timeout=None):
    """
    Run ElmerSolver, streaming its stdout live: every line is written to
    log_file AND scanned for Elmer's "Time: N/M" progress marker. Prints
    a step-progress + per-run ETA line to the console periodically
    (throttled by PROGRESS_PRINT_INTERVAL_SECONDS), without spamming
    the console on every single solver line.
    Returns True on success (exit code 0), False otherwise.
    """
    start_time = time.time()
    last_print_time = 0.0
    last_step, last_total = None, None

    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        log.flush()

        try:
            proc = subprocess.Popen(
                cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,  # line-buffered
            )
        except FileNotFoundError as e:
            log.write(f"\n!! COMMAND NOT FOUND: {e}\n")
            print(f"    !! COMMAND NOT FOUND: {e}")
            return False

        try:
            for line in proc.stdout:
                log.write(line)

                match = STEP_PATTERN.search(line)
                if match:
                    last_step, last_total = int(match.group(1)), int(match.group(2))

                now = time.time()
                if now - last_print_time >= PROGRESS_PRINT_INTERVAL_SECONDS:
                    elapsed = now - start_time
                    if last_step and last_total:
                        pct = 100.0 * last_step / last_total
                        avg_per_step = elapsed / last_step
                        remaining_steps = last_total - last_step
                        eta_seconds = avg_per_step * remaining_steps
                        print(
                            f"    ...step {last_step}/{last_total} ({pct:.1f}%) | "
                            f"elapsed {format_duration(elapsed)} | "
                            f"ETA (this run) {format_duration(eta_seconds)}"
                        )
                    else:
                        print(f"    ...running, elapsed {format_duration(elapsed)} "
                              f"(no step markers seen yet)")
                    last_print_time = now

                if timeout is not None and (time.time() - start_time) > timeout:
                    proc.kill()
                    log.write(f"\n!! TIMED OUT after {timeout}s\n")
                    print(f"    !! TIMED OUT after {format_duration(timeout)}")
                    return False

            proc.wait()

        except Exception as e:
            proc.kill()
            log.write(f"\n!! EXCEPTION while streaming output: {e}\n")
            print(f"    !! EXCEPTION while streaming output: {e}")
            return False

        if proc.returncode != 0:
            log.write(f"\n!! Exited with return code {proc.returncode}\n")
            print(f"    !! ElmerSolver exited with return code {proc.returncode}")
            return False

    total_elapsed = time.time() - start_time
    print(f"    ElmerSolver finished in {format_duration(total_elapsed)}.")
    return True


def patch_sif_mesh_db(sif_text, mesh_dir_name):
    """Rewrite the second quoted argument of the 'Mesh DB' line."""
    pattern = re.compile(r'(Mesh\s+DB\s+"[^"]*"\s+)"([^"]*)"', re.IGNORECASE)
    new_text, n_subs = pattern.subn(lambda m: f'{m.group(1)}"{mesh_dir_name}"', sif_text)
    if n_subs == 0:
        raise ValueError(
            "Could not find a 'Mesh DB \"...\" \"...\"' line in this SIF to patch."
        )
    if n_subs > 1:
        print(f"    WARNING: patched {n_subs} 'Mesh DB' lines (expected 1).")
    return new_text


def get_or_build_mesh_cache(msh_path):
    """
    Convert msh_path via ElmerGrid into MESH_CACHE_DIR/<msh_stem>/ once,
    reusing it on subsequent calls if REUSE_MESH_CONVERSION is True.
    Returns the path to the cached mesh_db-equivalent directory, or None
    on failure.
    """
    cache_dir = MESH_CACHE_DIR / msh_path.stem
    marker = cache_dir / ".conversion_ok"

    if REUSE_MESH_CONVERSION and marker.exists():
        print(f"    Mesh cache hit for {msh_path.name} — skipping ElmerGrid.")
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_msh = cache_dir / "mesh.msh"
    shutil.copy2(msh_path, local_msh)

    log_file = cache_dir / "elmergrid_log.txt"
    print(f"    Converting mesh (ElmerGrid) for {msh_path.name}...")
    t0 = time.time()
    ok = run_command_simple(
        [ELMERGRID_EXE, "14", "2", local_msh.name, "-out", "converted"],
        cwd=cache_dir,
        log_file=log_file,
    )
    print(f"    ElmerGrid finished in {format_duration(time.time() - t0)}.")

    converted_dir = cache_dir / "converted"
    if not ok or not converted_dir.exists():
        print(f"    FAILED to convert {msh_path.name}. See {log_file}")
        return None

    marker.write_text("ok")
    return converted_dir


def process_combination(sif_name, sif_template_text, msh_path, variant_index):
    variant_name = f"variant_{variant_index:02d}"
    variant_dir = OUTPUT_ROOT / sif_name / variant_name
    completed_marker = variant_dir / COMPLETED_MARKER_NAME

    # ── Already fully completed in a previous run? Skip entirely. ──
    if completed_marker.exists():
        print(f"\n  [{variant_name}] mesh = {msh_path.name} — already completed, skipping.")
        return True

    # ── Not completed (never started, or was interrupted). Wipe and redo cleanly. ──
    if variant_dir.exists():
        print(f"\n  [{variant_name}] mesh = {msh_path.name} — found incomplete folder "
              f"(no {COMPLETED_MARKER_NAME} marker), redoing this variant from scratch.")
        shutil.rmtree(variant_dir)
    else:
        print(f"\n  [{variant_name}] mesh = {msh_path.name}")

    variant_dir.mkdir(parents=True, exist_ok=True)

    log_file = variant_dir / "run_log.txt"
    log_file.write_text(
        f"=== {sif_name} / {variant_name} :: mesh = {msh_path.name} ===\n"
    )

    # ── Get this mesh's converted output (from cache, or convert fresh) ──
    converted_dir = get_or_build_mesh_cache(msh_path)
    if converted_dir is None:
        print(f"    SKIPPED — mesh conversion failed.")
        return False

    mesh_db_dir = variant_dir / MESH_DIR_NAME
    if mesh_db_dir.exists():
        shutil.rmtree(mesh_db_dir)
    shutil.copytree(converted_dir, mesh_db_dir, ignore=shutil.ignore_patterns(".conversion_ok"))

    # ── Patch and write this variant's case.sif ──
    patched_sif = patch_sif_mesh_db(sif_template_text, MESH_DIR_NAME)
    variant_sif_path = variant_dir / "case.sif"
    variant_sif_path.write_text(patched_sif, encoding="utf-8")

    # ── ELMERSOLVER_STARTINFO ──
    (variant_dir / "ELMERSOLVER_STARTINFO").write_text("case.sif\n1\n", encoding="utf-8")

    # ── Ensure results dir exists (per SIF's "Results Directory") ──
    (variant_dir / "results").mkdir(exist_ok=True)

    # ── Run ElmerSolver, with live step progress + ETA ──
    print(f"    Running ElmerSolver...")
    ok = run_solver_with_progress(
        [ELMERSOLVER_EXE], cwd=variant_dir, log_file=log_file,
        timeout=SOLVER_TIMEOUT_SECONDS,
    )
    if not ok:
        print(f"    FAILED. See {log_file}")
        return False

    completed_marker.write_text(datetime.datetime.now().isoformat())
    print(f"    Done -> {variant_dir / 'results'}")
    return True


def main():
    sif_files = find_files(SIF_INPUT_DIR, "*.sif")
    msh_files = find_files(MSH_INPUT_DIR, "*.msh")

    total_combinations = len(sif_files) * len(msh_files)
    print(f"Found {len(sif_files)} SIF files and {len(msh_files)} mesh files.")
    print(f"Total runs: {len(sif_files)} x {len(msh_files)} = {total_combinations}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MESH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    completed_count = 0
    combination_durations = []  # seconds, one entry per finished combination
    batch_start_time = time.time()

    for sif_path in sif_files:
        sif_name = sif_path.stem
        print(f"\n══ SIF: {sif_name} ══")
        sif_template_text = sif_path.read_text(encoding="utf-8")

        sif_results = {}
        for i, msh_path in enumerate(msh_files, start=1):
            combo_start = time.time()
            ok = process_combination(sif_name, sif_template_text, msh_path, i)
            combo_elapsed = time.time() - combo_start

            sif_results[msh_path.name] = ok
            completed_count += 1
            combination_durations.append(combo_elapsed)

            # ── Batch-level progress + ETA, based on running average ──
            avg_duration = sum(combination_durations) / len(combination_durations)
            remaining = total_combinations - completed_count
            batch_eta_seconds = avg_duration * remaining
            batch_elapsed = time.time() - batch_start_time

            print(
                f"    [BATCH PROGRESS] {completed_count}/{total_combinations} done "
                f"({100.0 * completed_count / total_combinations:.1f}%) | "
                f"batch elapsed {format_duration(batch_elapsed)} | "
                f"avg/run {format_duration(avg_duration)} | "
                f"batch ETA {format_duration(batch_eta_seconds)}"
            )

        summary[sif_name] = sif_results

    # ── Final report ──
    print("\n══════════════════════════════════════")
    print("BATCH SUMMARY")
    print("══════════════════════════════════════")
    total, total_ok = 0, 0
    for sif_name, results in summary.items():
        n_ok = sum(results.values())
        total += len(results)
        total_ok += n_ok
        print(f"  {sif_name}: {n_ok}/{len(results)} variants OK")
        for msh_name, ok in results.items():
            if not ok:
                print(f"      FAILED -> {msh_name}")

    total_batch_time = time.time() - batch_start_time
    print(f"\n{total_ok} / {total} total runs completed successfully.")
    print(f"Total batch time: {format_duration(total_batch_time)}")


if __name__ == "__main__":
    main()