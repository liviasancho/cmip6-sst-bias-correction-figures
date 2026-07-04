# CMIP6 SST Bias Correction: Supplementary Figures, Metadata, and Scripts

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20617602.svg)](https://doi.org/10.5281/zenodo.20617602)

This repository provides supplementary figures, metadata, selected input data, corrected SST time series, and Python scripts associated with the book chapter on sea surface temperature bias correction in CMIP6 models.

The material includes diagnostic figures, performance summaries, visual comparisons between original CMIP6 simulations, bias-corrected outputs, and the NOAA OISST V2 observational dataset, as well as scripts used for bias diagnostics and bias removal.

## Scope

The repository contains material related to:

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

- Daily NOAA OI SST V2 High Resolution Dataset, provided by the NOAA PSL, Boulder, Colorado, USA, available at https://psl.noaa.gov, 0.25°, 1981-2014.

## Bias correction methods

The evaluated methods include:

- Linear Regression
- Empirical Quantile Mapping
- XGBoost

## Repository structure

- `figures/bias_diagnostics/`: diagnostic figures used to evaluate SST bias behavior
- `figures/bias_removal/`: figures comparing original and bias-corrected SST series
- `input_data/`: input files needed to run the Python scripts
- `metadata/`: figure index and auxiliary metadata files
- `out_sst/`: corrected SST time series for each model, region, experiment, and method
- `scripts/`: Python scripts used for bias diagnostics and bias removal
- `tables/`: performance metrics, method selection tables, and improvement summaries

## Scripts

The repository includes two main scripts:

- `scripts/bias_diagnostics.py`: performs the SST bias diagnostic analysis and generates diagnostic figures and tables.
- `scripts/bias_removal.py`: applies the bias correction methods, evaluates performance metrics, saves corrected SST time series, and generates comparison figures.

## Environment

The recommended way to reproduce the Python environment is using Conda:

```bash
conda env create -f environment.yml
conda activate cmip6-bias-removal

## How to use this repository

The file `metadata/figure_index.csv` provides a searchable catalog of the figures, including model, region, experiment, method, figure type, period, and short description.

To run the scripts, users should check the expected input files in `input_data/` and adjust local paths if necessary.

## Citation

If you use this repository, please cite the archived version:

Sancho, Lívia; da Fonseca Aguiar, Louise; Galves, Vitor Luiz Victalino; Coutinho, Priscila Esposte; Guida, Aimée; Cataldi, Marcio. 2026. _CMIP6 SST Bias Correction: Supplementary Figures, Metadata, and Scripts_. Zenodo. https://doi.org/10.5281/zenodo.20617602

## License

Figures, tables, metadata, and documentation are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

Code in the `scripts/` directory is provided for transparency and reproducibility. If reused or adapted, please cite this repository.
