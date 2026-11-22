# Changelog

## v2.0.0 — revised article release

Accompanies the revised manuscript and supplement (title, scope and several analyses changed in response to peer review).

### Added
- `analysis/check_public_results.py`: reconstructs the 12 paired period-contrast summaries and the factorial terms from the public tables, without locality data.
- `analysis/review_diagnostics.py`: active-predictor MESS comparison, response at common predictor values, paired P10 resampling partition.
- `results/sensitivity/`: factorial cells/terms/shares, resampled areas, threshold contrasts, BIO12 common-environment check, 25- vs 30-year recent-period comparison.
- `results/diagnostics/`: full tuning grid, selected settings, paired AUC equivalence, contrasts with CIs, robustness table.
- Supplementary tables S23–S29.
- `.zenodo.json`, `.gitattributes`.

### Changed
- README and CITATION.cff rewritten for the revised title and scope.
- Updated Tables 2–3 and Tables S2, S8, S9, S16 and step 13 (tuning and spatial cross-validation) and `evaluation.py`.

### Unchanged
- Study design: four calibrations (WorldClim/TerraClimate × 1970–1999/2000–2024), 38 thinned localities, eight predictors.
- Licences: MIT (code), CC BY 4.0 (derived tables and documentation).

Previous release: v1.0.1 (original submission), https://doi.org/10.5281/zenodo.21539728
