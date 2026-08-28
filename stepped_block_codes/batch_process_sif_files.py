"""
Generates Elmer .sif case files for all combinations of
pouring temperature and mould temperature.

Output file naming: case_<pouringtemp>_<mouldtemp>.sif
Example: case_953_303.sif
"""

import os

# ── Parameter sweep ─────────────────────────────────────────────────────────
pouring_temperatures = [953, 973, 993, 1013, 1033]   # K
mould_temperatures   = [303, 353, 403]                # K

# ── Output directory ────────────────────────────────────────────────────────
output_dir = "block_sif_cases"
os.makedirs(output_dir, exist_ok=True)

# ── SIF Template ─────────────────────────────────────────────────────────────
# {pour} and {mould} are placeholders replaced per combination.
# Original baseline values (for reference, used in comments):
#   Pouring temperature baseline = 993.0
#   Mould temperature baseline   = 293.0

sif_template = """Header
  CHECK KEYWORDS Warn
  Mesh DB "." "Stepped_Block"
  Include Path ""
  Results Directory "results"
End

! ─── SIMULATION SETUP ────────────────────────────────────────────────────────
Simulation
  Max Output Level = 5
  Coordinate System = String "Cartesian"
  Coordinate Mapping(3) = 1 2 3

  ! Converts Gmsh millimeters to Elmer meters
  Coordinate Scaling = Real 0.001
  Simulation Type = String "Transient"

  Timestepping Method = String "BDF"
  BDF Order = 2

  Steady State Max Iterations = 5

  Timestep intervals(4) = 600   400   400   400
  Timestep Sizes(4)     = 0.1   0.5   2.0   20.0
End

Constants
  Gravity(4) = 0 0 -1 9.81
  Stefan Boltzmann = 5.67e-08
End

! ─── BODY ────────────────────────────────────────────────────────────────────
! Target Bodies updated: mesh.names maps "Block" -> physical body index 2
! (was 184 for the impeller mesh)
Body 1
  Target Bodies(1) = 2
  Name = String "body1"
  Equation = 1
  Material = 1
  Initial Condition = 1
  ! NOTE: Body Force removed — it only existed to declare Boussinesq
  ! buoyancy for Navier-Stokes, which is no longer solved.
End

! ─── INITIAL CONDITIONS ──────────────────────────────────────────────────────
Initial Condition 1
  Name = String "Freshly_Poured_Mold"
  ! value changed 993.0 --> {pour}.0  (Pouring temperature of LM6 Aluminum)
  Temperature = Real {pour}.0
End

! ─── MATERIAL: LM6 / AlSi12 ──────────────────────────────────────────────────
! Phase temperatures (BS 1490 / ASM Handbook Vol.15):
!   Solidus  = 850 K (577 °C) — eutectic reaction temperature
!   Liquidus = 886 K (613 °C)
!   Latent heat of fusion ~ 390 kJ/kg, released predominantly near the
!   eutectic point due to LM6's near-eutectic composition
!
Material 1
  Name = String "LM6_Aluminum_Alloy"
  Density = Real 2650.0
  Reference Temperature = Real 933.0     ! Midpoint of mushy zone (K)
  ! NOTE: Heat Expansion Coefficient removed — only used by Boussinesq
  ! buoyancy in Navier-Stokes, which is no longer solved.

 ! ── Enthalpy ─────────────────────────────────────────
  ! Enthalpy contains ONLY latent heat (390 kJ/kg)
  ! released linearly across mushy zone 850-886 K
  ! Sensible heat is handled separately by Heat Capacity above
  Enthalpy = Variable Temperature
    Real
      293.0      0.0
      849.9      0.0      ! Zero outside mushy zone
      850.0      0.0      ! Solidus / eutectic — latent heat starts
      886.0  390000.0     ! Liquidus — full 390 kJ/kg released
      1000.0 390000.0     ! Constant above liquidus
    End

 ! ── Specific heat capacity ─────────────────────────────────────────
  Heat Capacity = Variable Temperature
    Real
      293.0    871.0
      473.0    920.0
      673.0    963.0
      850.0   1050.0
      886.0   1080.0
      1000.0  1080.0
    End

  ! ── Thermal Conductivity (W/m·K) ─────────────────────────────────────────
  ! ASM Handbook Vol.15 / NPL Mills (2002) values for AlSi12
  Heat Conductivity = Variable Temperature
    Real
      293.0   155.0    ! Solid - ASM Vol.15
      473.0   159.0
      673.0   163.0
      850.0   100.0    ! Solidus - drops sharply as eutectic dissolves
      886.0    90.0    ! Liquidus - fully molten
      1000.0   90.0
    End

  ! NOTE: Viscosity table removed — only used by Navier-Stokes, which is
  ! no longer solved.
End

! ─── EQUATION ────────────────────────────────────────────────────────────────
! Only Solver 1 (Heat) and Solver 2 (Flux/Gradient), 3 (Output) remain
! active — Navier-Stokes solver removed and renumbered out.
Equation 1
  Name = String "Thermal_Only"
  Active Solvers(3) = 1 2 3
  Convection = String "None"
  Phase Change Model = String "Spatial 2"
End

! ─── SOLVER 1: HEAT EQUATION ─────────────────────────────────────────────────
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

! ─── SOLVER 2: FLUX & GRADIENT ───────────────────────────────────────────────
! (was Solver 3 in the coupled version — renumbered since NS is removed)
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

! ─── SOLVER 3: VTU OUTPUT ────────────────────────────────────────────────────
! (was Solver 4 in the coupled version — renumbered since NS is removed)
Solver 3
  Equation = String "ResultOutput"
  Procedure = File "ResultOutputSolve" "ResultOutputSolver"
  Exec Solver = String "After Timestep"
  Output Format = String "Vtu"
  Output File Name = File "block_data"
  Save Geometry Ids = Logical True
  Scalar Field 1 = String "Temperature"
 ! Vector Field 1 = String "Temperature Grad"

End

! ─── BOUNDARY CONDITION 1 — BLOCK SURFACE ──────────────────────────────────
Boundary Condition 1
  Target Boundaries(1) = 1
  Name = String "Block_Surface"
  ! value changed 293.0 --> {mould}.0  (Sand mould temperature)
  External Temperature = Real {mould}.0
  Heat Transfer Coefficient = Variable Temperature
    Real
      ! value changed 293.0 --> {mould}.0  (lowest point now matches new mould temperature)
      {mould}.0    35.0
      850.0   110.0
      868.0   300.0
      886.0   633.3
      ! value changed 993.0 --> {pour}.0  (highest point now matches new pouring temperature)
      {pour}.0   633.3
    End
End
"""

# ── Generate all 15 combinations ─────────────────────────────────────────────
count = 0
for pour in pouring_temperatures:
    for mould in mould_temperatures:
        sif_content = sif_template.format(pour=pour, mould=mould)

        filename = f"case_{pour}_{mould}.sif"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(sif_content)

        count += 1
        print(f"Generated: {filepath}")

print(f"\nDone. {count} .sif files generated in '{output_dir}/'")