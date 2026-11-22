# Sensitivity of MaxEnt suitability projections for *Hemiscorpius acanthocercus* in Iran to climate product and calibration period

**Analysis code and derived results · v2.0.0**

This release accompanies the revised article by Marjan Ahangarha and co-authors. Four MaxEnt calibrations cross WorldClim and TerraClimate with 1970–1999 and 2000–2024. They share 38 spatially thinned localities, eight predictors and common settings. The study compares spatial discrimination, projected area and extrapolation. Separate sensitivity analyses examine published WorldClim normals, annual precipitation from CHIRPS, tuning and threshold estimation. The article and its supplement give the ecological interpretation.

## Find a result

| Question | Files |
|---|---|
| Which inputs, predictors and settings were used? | [Input sources](results/tables/Table_1_01.csv), [predictors](results/tables/Table_2_01.csv), [screening decisions](results/tables/Table_S3_01.csv) and [tuning grid](results/diagnostics/13_tuning_all_settings.csv) |
| How do the validation scores and projected areas compare? | [Spatial validation](results/tables/Table_3_01.csv), [paired AUC comparisons](results/diagnostics/T17_A_equivalence_auc.csv), [areas](results/tables/Table_4_01.csv) and [variance components](results/tables/Table_5_01.csv) |
| How sensitive are projections to the fitting choices? | [Alternative specifications](results/tables/Table_S16_01.csv), [thresholds](results/sensitivity/V4_threshold_contrasts.csv), [additional calibrations](results/tables/Table_S28_01.csv) and [factorial results](results/sensitivity/) |
| How do I check a result without the locality data? | Run the command below from the repository root. |

Install NumPy and pandas in a Python environment, then run:

```sh
python analysis/check_public_results.py
```

The check reconstructs the twelve paired period-contrast summaries from 720 area records and the 63 factorial terms for each of two area measures from 160 design cells. It uses temporary output files. All areas are in km². P10 is the tenth-percentile training-presence threshold, MTP the minimum training-presence threshold and maxSSS the maximum sum of sensitivity and specificity. In the resampling files, `area` uses each replicate's own cutoff and `fixed_area` uses its full-fit cutoff. Table filenames retain the numbering of the manuscript and supplement. A final suffix such as `_02` identifies the second table block. The individual locality inventory in Table S1 is outside this archive's aggregate-results scope.

## Run the spatial analysis with the required inputs

1. Use a private working copy. Install the Python dependencies from `workflow/requirements.txt`; `workflow/requirements-lock.txt` records the project environment. Provide Java, MaxEnt 3.4.4 at `workflow/maxent/maxent.jar`, and R with raster, dismo and spThin. The input catalog and resolutions are in Table 1. Geodesic terrain derivatives were prepared from the 30 m DEM before aggregation.
2. From `workflow/`, set the raw environmental data root and Rscript executable in `config/config.yaml` for your installation. Supply the named environmental files and the authorised occurrence input. The numbered Python scripts 01 through 19 document the processing sequence, including step 10b. Steps 04 and 09 call their R helpers. Pass the --all flag to Python steps 13, 14, 15 and 16 to include the sensitivity variants required by the later comparisons. The configuration specifies the four baselines, random seed, resampling, accessible area and tuning grid. For the paired AUC comparison, step 17 uses a mean of four within-block comparisons.
3. Once the processed inputs, fits, response curves and projections are available, run `python analysis/revision_analysis.py`, `python analysis/factorial.py`, `python analysis/verify_results.py` and `python analysis/partition_interactions.py` from the repository root. The additional analysis writes its tables under `analysis/generated/` and its fits under `analysis/fits/`. Intermediate training files and figures created during a private rerun can contain localities.

4. After rebuilding the processed inputs and projections in the private workflow, run `python analysis/review_diagnostics.py workflow results/sensitivity` from the repository root. This writes the active-predictor MESS comparison, the response at common predictor values, and the paired P10 resampling partition. The aggregate tables are also supplied in `results/sensitivity/`, alongside the comparison of 25- and 30-year recent climatologies.

## Access, citation and rights

Environmental datasets are available from the original providers listed in Table 1 under their own terms. Requests for controlled access to occurrence coordinates and specimen information may be sent to the corresponding author under the authors’ data-use conditions. The public release contains analysis source and derived tables. Full model fitting requires the environmental layers and authorised locality data.

Cite the authors in `CITATION.cff`, the software title, version and the version-specific DOI shown on its Zenodo record. [Zenodo version history](https://doi.org/10.5281/zenodo.21539727) · [Original submission release, v1.0.1](https://doi.org/10.5281/zenodo.21539728) · [Revised release, v2.0.0](https://doi.org/10.5281/zenodo.24448818).

The authors' source code is available under the [MIT License](LICENSE). The authors' derived tables and original documentation are available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
