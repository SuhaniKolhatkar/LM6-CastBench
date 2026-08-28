# -*- coding: utf-8 -*-
"""
batch_process_sif_files.py  (final - full 15-case sweep, calibrated schedule)

Generates the full 15-case process-parameter sweep of Elmer .sif files
for the flange (variant 1 geometry, unchanged), using a single shared
timestepping schedule that was validated against the two extreme
corners of the sweep.

CALIBRATION RESULT (from the 2 extreme-corner test runs)
------------------------------------------------------------------------
  pour=953 K,  mould=403 K (smallest driving Delta T = 550 K):
      residual at t=2000s -> 0.22 K from mould temp
      residual at t=2400s -> 0.06 K from mould temp

  pour=1033 K, mould=303 K (largest driving Delta T = 730 K):
      residual at t=2000s -> 1.11 K from mould temp
      residual at t=2400s -> 0.39 K from mould temp

Both extreme corners are comfortably converged (< 0.5 K residual) by
t=2400s. Since these are the mathematical extremes of the 5x3 sweep,
every other combination settles at least as fast - so ONE shared
schedule is valid for all 15 cases; no per-case tailoring needed.

LOCKED-IN SHARED SCHEDULE (Stage 4 extended from 60 -> 80 steps vs. the
single-geometry reference, to buy the extra convergence margin found
above):
  Stage 1 (0-60s):     dt=0.1s,  600 steps  - pour + eutectic crossing
  Stage 2 (60-300s):   dt=1.0s,  240 steps  - steep post-eutectic decline
  Stage 3 (300-800s):  dt=5.0s,  100 steps  - moderating decline
  Stage 4 (800-2400s): dt=20.0s,  80 steps  - final approach to mould
                        temperature (extended from 60->80 steps /
                        2000s->2400s based on calibration above)
  Total: 1020 steps, 2400 s simulated time - same schedule for all 15 cases.

PARAMETER SWEEP
----------------
pouring_temperatures = [953, 973, 993, 1013, 1033]  K  (5 values)
mould_temperatures   = [303, 353, 403]               K  (3 values)
-> 5 x 3 = 15 combinations, one .sif file each.

WHAT CHANGES PER CASE
-----------------------
1. Initial Condition 1 -> Temperature = pouring_temperature
2. Boundary Condition 1 -> External Temperature = mould_temperature
Material property table domains are widened to 1060 K (values
unchanged) so the highest pour temperature (1033 K) stays inside the
tabulated range. Everything else (alloy physics, mesh reference,
solver settings, the shared schedule above) is identical across all
15 files.

Output: flange_sif_cases/flange_pour<P>_mould<M>.sif  (15 files)
        flange_sif_cases/sif_sweep_summary.csv

Run with:
    python batch_process_sif_files.py
"""

import os
import csv
import itertools

# -----------------------------------------------------------------
# Parameter sweep
# -----------------------------------------------------------------
pouring_temperatures = [953, 973, 993, 1013, 1033]   # K
mould_temperatures   = [303, 353, 403]                # K

OUT_DIR = r"D:\NeurIPS\surrogate_flange\flange_sif_cases"
MESH_DIR = "mesh_db"   # same geometry for the whole sweep


# -----------------------------------------------------------------
# .sif template (calibrated shared schedule)
# -----------------------------------------------------------------
SIF_TEMPLATE = """Header
  CHECK KEYWORDS Warn
  Mesh DB "." "{mesh_dir}"
  Include Path ""
  Results Directory "results"
End

! ============================================================================
! FLANGE VARIANT 1 - PROCESS PARAMETER SWEEP CASE
!   Pouring temperature = {pour_temp} K
!   Mould temperature   = {mould_temp} K
! ============================================================================
! Same validated geometry, mesh, and alloy physics as the reference
! flange_1.sif. Two physical inputs differ for this case:
!
!   1. Initial Condition 1 -> Temperature = {pour_temp}.0
!      (melt temperature at the moment the mould cavity is filled)
!
!   2. Boundary Condition 1 -> External Temperature = {mould_temp}.0
!      (mould preheat temperature - the casting cools toward this,
!      not toward plain room temperature)
!
! Material property tables (Enthalpy/Heat Capacity/Heat Conductivity)
! and the Heat Transfer Coefficient curve have their upper temperature
! breakpoint widened to 1060 K (values unchanged, domain only) so the
! highest pouring temperature in this sweep (1033 K) stays inside the
! tabulated range.
!
! TIMESTEPPING - CALIBRATED SHARED SCHEDULE (validated against the two
! extreme corners of this sweep: pour953/mould403 and pour1033/mould303
! both converge to within 0.4 K of their target mould temperature by
! t=2400s). Every other combination in this sweep sits between these
! two extremes and settles at least as fast, so this single schedule
! is used for all 15 cases without per-case tailoring:
!   Stage 1 (0-60s):     dt=0.1s,  600 steps
!   Stage 2 (60-300s):   dt=1.0s,  240 steps
!   Stage 3 (300-800s):  dt=5.0s,  100 steps
!   Stage 4 (800-2400s): dt=20.0s,  80 steps
! ============================================================================

Simulation
  Max Output Level = 5
  Coordinate System = String "Cartesian"
  Coordinate Mapping(3) = 1 2 3

  Coordinate Scaling = Real 0.001
  Simulation Type = String "Transient"

  Timestepping Method = String "BDF"
  BDF Order = 2

  Steady State Max Iterations = 5

  Timestep intervals(4) = 600   240   100   80
  Timestep Sizes(4)     = 0.1   1.0   5.0   20.0
End

Constants
  Gravity(4) = 0 0 -1 9.81
  Stefan Boltzmann = 5.67e-08
End

! --- BODY -------------------------------------------------------------------
Body 1
  Target Bodies(1) = 2
  Name = String "body1"
  Equation = 1
  Material = 1
  Initial Condition = 1
End

! --- INITIAL CONDITIONS ------------------------------------------------------
Initial Condition 1
  Name = String "Freshly_Poured_Flange"
  Temperature = Real {pour_temp}.0
End

! --- MATERIAL: LM6 / AlSi12 ---------------------------------------------------
Material 1
  Name = String "LM6_Aluminum_Alloy"
  Density = Real 2650.0
  Reference Temperature = Real 933.0

  Enthalpy = Variable Temperature
    Real
      293.0      0.0
      849.9      0.0
      850.0      0.0
      886.0  390000.0
      1060.0 390000.0
    End

  Heat Capacity = Variable Temperature
    Real
      293.0    871.0
      473.0    920.0
      673.0    963.0
      850.0   1050.0
      886.0   1080.0
      1060.0  1080.0
    End

  Heat Conductivity = Variable Temperature
    Real
      293.0   155.0
      473.0   159.0
      673.0   163.0
      850.0   100.0
      886.0    90.0
      1060.0   90.0
    End
End

! --- EQUATION -----------------------------------------------------------------
Equation 1
  Name = String "Thermal_Only"
  Active Solvers(3) = 1 2 3
  Convection = String "None"
  Phase Change Model = String "Spatial 2"
End

! --- SOLVER 1: HEAT EQUATION ---------------------------------------------------
Solver 1
  Equation = String "Heat Equation"
  Variable = String "Temperature"
  Procedure = File "HeatSolve" "HeatSolver"

  Linear System Solver = String "Iterative"
  Linear System Iterative Method = String "BiCGStab"
  Linear System Max Iterations = 500
  Linear System Convergence Tolerance = 1.0e-6
  Linear System Preconditioning = String "ILU1"
  Linear System Abort Not Converged = Logical False
  Linear System Residual Output = Integer 0

  Nonlinear System Max Iterations = 20
  Nonlinear System Convergence Tolerance = 1.0e-5
  Nonlinear System Newton After Iterations = 8
  Nonlinear System Newton After Tolerance = 1.0e-3
End

! --- SOLVER 2: FLUX & GRADIENT ---------------------------------------------------
Solver 2
  Equation = String "Flux and Gradient"
  Procedure = File "FluxSolver" "FluxSolver"
  Exec Solver = String "After Timestep"
  Calculate Grad = Logical True
  Target Variable = String "Temperature"
  Linear System Solver = String "Iterative"
  Linear System Iterative Method = String "CG"
  Linear System Preconditioning = String "ILU0"
  Linear System Max Iterations = 500
  Linear System Convergence Tolerance = 1.0e-7
End

! --- SOLVER 3: VTU OUTPUT ---------------------------------------------------------
Solver 3
  Equation = String "ResultOutput"
  Procedure = File "ResultOutputSolve" "ResultOutputSolver"
  Exec Solver = String "After Timestep"
  Output Format = String "Vtu"
  Output File Name = File "flange_pour{pour_temp}_mould{mould_temp}_data"
  Save Geometry Ids = Logical True
  Scalar Field 1 = String "Temperature"
End

! --- BOUNDARY CONDITIONS -----------------------------------------------------
Boundary Condition 1
  Target Boundaries(1) = 1
  Name = String "Flange_Surface"
  External Temperature = Real {mould_temp}.0
  Heat Transfer Coefficient = Variable Temperature
    Real
      293.0    35.0
      850.0   110.0
      868.0   300.0
      886.0   633.3
      1060.0  633.3
    End
End
"""


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
def main():
    out_dir = os.path.abspath(OUT_DIR)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    combinations = list(itertools.product(pouring_temperatures, mould_temperatures))
    print("Generating {} .sif files ({} pour temps x {} mould temps) ...".format(
        len(combinations), len(pouring_temperatures), len(mould_temperatures)))

    summary_rows = []

    for pour_temp, mould_temp in combinations:
        sif_text = SIF_TEMPLATE.format(
            mesh_dir=MESH_DIR,
            pour_temp=pour_temp,
            mould_temp=mould_temp,
        )

        filename = "case_{}_{}.sif".format(pour_temp, mould_temp)
        out_path = os.path.join(out_dir, filename)

        # encoding="utf-8" avoids UnicodeEncodeError on Windows, where
        # open() otherwise defaults to the system codepage (cp1252).
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(sif_text)

        print("  pour={} K, mould={} K -> {}".format(pour_temp, mould_temp, out_path))

        summary_rows.append({
            "pouring_temperature_K": pour_temp,
            "mould_temperature_K": mould_temp,
            "delta_T_K": pour_temp - mould_temp,
            "sif_file": filename,
        })

    report_path = os.path.join(out_dir, "sif_sweep_summary.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print("\nDone. {} .sif files written to {}".format(len(combinations), out_dir))
    print("Summary: {}".format(report_path))


if __name__ == "__main__":
    main()