# Supporting code for the dual-baseline SDM analysis of *Hemiscorpius acanthocercus*

This repository contains supporting analysis scripts for the manuscript on climatic-baseline choice and temporal mismatch in species distribution modelling of *Hemiscorpius acanthocercus* in Iran.

The code documents the main computational steps used for environmental-layer preparation, predictor screening, MaxEnt model tuning, binary thresholding, future-climate ensemble preparation, MESS/MoD analysis, habitat-area and change summaries, and spatial agreement metrics.

## Important note on workflow scope

The workflow was not fully automated in Python. Some final spatial-processing, visual inspection, cartographic layout, map production, and GIS-based checking steps were performed in ArcGIS Desktop 10.8 and/or ArcGIS Pro. These steps are summarized in `docs/arcgis_workflow_notes.md`.

The scripts should therefore be cited and interpreted as supporting analysis scripts for the manuscript, not as a complete replacement for the GIS workflow used to prepare the final maps.

## MaxEnt validation setting

Users should edit file paths at the top of each script before running.
