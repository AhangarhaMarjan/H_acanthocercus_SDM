# Code folder

This folder contains the supporting analysis scripts used for the dual-baseline species distribution modelling workflow of *Hemiscorpius acanthocercus* in Iran.

The scripts are provided to document and support the analyses reported in the manuscript. They do not include raw dissertation data, raw field notebooks, specimen-level records, or analyses outside the scope of the manuscript.

## Important reproducibility note

The workflow was not fully automated in Python. Several spatial-processing and cartographic steps were performed in ArcGIS Desktop/ArcGIS Pro, including raster alignment checks, masking, map layout production, visual inspection, and final cartographic editing. These GIS-based steps are documented in `docs/arcgis_workflow_notes.md`.

The Python scripts should therefore be read as the supporting computational workflow for predictor preparation, statistical screening, MaxEnt command-line execution or reproduction of MaxEnt settings, thresholding, MESS/MoD calculations, area/change summaries, and agreement metrics. They are not intended to replace the GIS workflow used for final map preparation.

## Suggested running order

1. `01_standardize_ascii_headers_and_nodata.py`
2. `02_export_rasters_to_ascii_for_maxent.py`
3. `03_normality_tests_predictors.py`
4. `04_spearman_correlation_filtering.py`
5. `05_vif_predictor_selection.py`
6. `12_run_maxent_tuning_and_final_models.py`
7. `06_threshold_and_create_binary_maps.py`
8. `08_future_predictor_ensemble_variables.py`
9. `07_future_maxent_ensemble_mean.py`
10. `09_calculate_mess_and_mod.py`
11. `10_area_change_and_habitat_dynamics.py`
12. `11_spatial_agreement_and_transferability_metrics.py`

## Before running

Edit input and output paths at the top of each script. The paths are intentionally written as placeholders to avoid storing local computer paths in the public repository.
