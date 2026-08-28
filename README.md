# LM6-CastBench

Code accompanying **LM6-CastBench: A Geometry-Stratified Simulation Dataset and Benchmark for Transient Thermal Surrogate Modelling in Aluminum Sand Casting**.

This repository contains the full data-generation and benchmarking pipeline used to build the dataset, organized by geometry family:

- `brake_codes/` — pipeline for the brake lever geometry
- `flange_codes/` — pipeline for the flange geometry
- `stepped_block_codes/` — pipeline for the stepped block geometry

Each folder mirrors the same end-to-end workflow described in the paper, applied to that geometry family: parametric CAD generation and Latin-hypercube variant sampling, mesh generation, transient thermal simulation in Elmer FEM, post-processing of raw solver output into model-ready tensors, and the MLP / U-Net surrogate training and benchmarking scripts used to produce the reported results.

## Dataset

The full simulation dataset is hosted on Hugging Face:

**Hugging Face dataset:** <!-- PASTE LINK HERE -->

## Citation

If you use this code or dataset, please cite the accompanying paper.
