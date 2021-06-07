# Probabilistic Substance Quality Index (PSQI)

This repository contains experimental results and source code for the paper that introduced PSQI index and applied it to water quality measurements conducted in New Moscow region, Russia.

# Results

## Training data (results/data.zip/data.csv)

Contains filtered (outliers and some non-ifnormative parameters removed) original measurements at locations defined by latitude and longitude. Additionally, it includes computed PSQI, its confidence values and "honest" quality index. This index is computed directly as an average number of the measured parameters that lie in admissible bounds defined by expert knowledge and/or government standards.

## Experimental results (results/data.zip/results.csv)

Contains predicted values of parameters (mean, 1st and 99th percentiles) over the region with 100m resolution including PSQI and its confidence.

## Spatial map (results/map.json)

This interactive map of measurements can be opened with [Kepler.gl](https://kepler.gl/demo) service. To do this
1. Follow the [link](https://kepler.gl/demo);
2. Click "Add data" (if modal window not opened already);
3. Select "Load Map using URL";
4. Paste [this URL](https://raw.githubusercontent.com/tzoiker/psqi/master/results/map.json) of the raw map.json content file and click "Fetch".
Alternatively, one may download map.json file and upload it directly after the step 2.

# Source code

* `src/data` - raw data used for experiments;
* `src/psqi` - main code used for experiments;
* `src/results` - output of the code execution;
* `src/playground.ipynb` - jupyter notebook for interactive execution of the code.

## Preparation

1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
2. Create python environment with `conda create -n psqi python=3.6`.
3. Install gpytorch with `conda install --name psqi gpytorch==0.3.6 -c gpytorch`.
4. (Optional) To enable GPU support, install corresponding [pytoch](https://pytorch.org/get-started/previous-versions/#v180) version.
5. Activate environment with `conda activate psqi`.
6. Install other dependencies with `pip install -r requirements.txt`.
7. Install jupyter notebook kernel with `python -m ipykernel install --name=psqi`.

## Running the code

Run jupyter notebook with `jupyter notebook` and open `src/playground.ipynb` in the automatically opened browser window.
Now jupyter cells can be sequentially executed to reproduce the results.
