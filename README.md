# Dual-baseline SDM analysis of *Hemiscorpius acanthocercus* in Iran

This repository provides supporting scripts and workflow documentation associated with the manuscript:

**Climatic-baseline choice and temporal mismatch in species distribution models: effects on habitat suitability and transferability for *Hemiscorpius acanthocercus* (Scorpiones: Hemiscorpiidae) in Iran**

## Repository contents

- `code/`  
  Supporting Python scripts used for environmental-layer preparation, predictor screening, raster conversion, binary-map generation, future-climate ensemble summaries, MESS/MoD summaries, habitat-dynamics calculations, and spatial agreement metrics.

- `docs/`  
  Workflow notes describing spatial-processing, raster inspection, map production, and ArcGIS-related steps.

- `data/`  
  Data-availability notes. Raw occurrence records, specimen-level coordinates, field notes, and unpublished dissertation datasets are not publicly redistributed.

- `metadata/`  
  Metadata describing environmental data sources, final predictors, software, and workflow scope.

- `results/`  
  Results-availability notes. Detailed derived result tables and large raster outputs are not publicly redistributed at this stage.

## Data scope

This repository is intended to support transparency of the analytical workflow reported in the manuscript while protecting unpublished data from an ongoing PhD dissertation.

The raw occurrence database, specimen-level coordinates, field notes, and unpublished dissertation datasets are not publicly redistributed in this repository.

Processed occurrence data may be made available from the corresponding author upon reasonable request, subject to author approval and data-use conditions.

## Environmental data

Raw environmental raster datasets are not redistributed in this repository because they are publicly available from their original providers, including:

- TerraClimate
- WorldClim / CMIP6
- SoilGrids
- Copernicus DEM GLO-30

## Workflow note

The scripts provided here support the main computational analyses reported in the manuscript. The workflow was not designed as a fully automated end-to-end pipeline. Some final spatial processing, raster inspection, visual checking, cartographic layout, and map production steps were performed in ArcGIS Desktop 10.8.

## Software

The main tools used in this study include:

- MaxEnt 3.4.4
- Python 3.x
- ArcGIS Desktop 10.8
- R

## Citation

Please cite the associated manuscript when using or referring to this repository.