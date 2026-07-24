# Environmental data sources

This file summarizes the environmental datasets used in the dual-baseline species distribution modeling analysis of Hemiscorpius acanthocercus in Iran.

## Climatic data

### TerraClimate

TerraClimate monthly minimum temperature, maximum temperature, and precipitation data were used to build the contemporary climatic baseline for the period 2000–2024.

Monthly layers were processed to derive bioclimatic predictors representing the contemporary baseline used for MaxEnt calibration and projection.

### WorldClim v2.1

WorldClim v2.1 bioclimatic variables representing the 1970–2000 historical baseline were used as the conventional historical climatic baseline.

### WorldClim/CMIP6 future projections

Future bioclimatic projections were used for the 2041–2060 period under SSP2-4.5 and SSP5-8.5 scenarios.

The future climate predictors were based on a multi-model ensemble of selected GCMs.

## Topographic data

### Copernicus DEM GLO-30

Copernicus DEM GLO-30 was used to derive topographic predictors. In the final model, slope was retained as a topographic predictor.

## Soil data

### SoilGrids 2.0

SoilGrids 2.0 data at 0–5 cm depth were used to derive soil predictors. In the final model, clay content was retained as an edaphic predictor.

## Final predictors used in the model

The final predictor set used in the dual-baseline MaxEnt models included:

- Bio3: Isothermality
- Bio4: Temperature seasonality
- Bio8: Mean temperature of the wettest quarter
- Bio12: Annual precipitation
- Clay: Soil clay content
- Slope: Terrain slope

## Data redistribution note

Raw environmental raster datasets are not redistributed in this repository because they are publicly available from their original providers. This repository provides supporting scripts and workflow documentation for the analyses reported in the manuscript.