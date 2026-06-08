# CMIP6 SST Bias Correction Figures

This repository provides supplementary figures and metadata associated with the book chapter on sea surface temperature bias correction in CMIP6 models.

The material includes diagnostic figures, performance summaries, and visual comparisons between original CMIP6 simulations, bias-corrected outputs, and the NOAA OISST V2 observational dataset.

## Scope

The repository contains figures related to:

- Historical experiment: 1981-2014
- SSP2-4.5 experiment: 2015-2050
- SSP5-8.5 experiment: 2015-2050

The analyzed Earth System Models are:

- ACCESS-CM2
- BCC-CSM2-MR
- CMCC-ESM2
- EC-Earth3-Veg
- INM-CM5-0
- MIROC6

The observational reference is:

- Daily NOAA OI SST V2 High Resolution Dataset data provided by the NOAA PSL, Boulder, Colorado, USA, from their website at https://psl.noaa.gov, 0.25°, 1981-2014

## Bias correction methods

The evaluated methods include:

- Linear Regression
- Empirical Quantile Mapping
- XGBoost

## Repository structure

- figures/: supplementary figures organized by bias diagnostics and bias removal
- tables/: performance metrics, method improvement, and classification tables
- metadata/: figure index and auxiliary metadata files

## How to use this repository

The file metadata/figure_index.csv provides a searchable catalog of all figures, including model, region, experiment, method, figure type, period, and short description.

## Citation

If you use this material, please cite the associated book chapter and this repository.

A DOI will be provided after repository archiving through Zenodo.

## License

Figures, tables, and metadata in this repository are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

If code is added to this repository in the future, it should be licensed separately under an appropriate open-source software license, such as MIT or BSD-3-Clause.
