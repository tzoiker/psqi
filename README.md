# Probabilistic Substance Quality Index (PSQI)

This repository contains experimental results for the paper that introduced PSQI index and applied it to water quality measurements conducted in New Moscow region, Russia.

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
