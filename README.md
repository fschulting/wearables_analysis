# Heart Rate analysis of different wearables

This repository contains the code and Jupyter notebook with all generated plots, results and sample data for my minor research project:
'Exploring Health Insights from Polar Vantage V3, Polar Verity Sense and Polar H10 Integration within the RADAR-base Platform.'

## Project

This internship project was conducted at [The Hyve](https://www.thehyve.nl/), in the RADAR-base team and involved the implementataion of different Polar devices to the [RADAR-base platform](https://github.com/RADAR-base).

The objective of this study was to assess the added value of three newly integrated Polar devices — the Polar Verity Sense, the Polar Vantage V3, and the Polar H10 — to the RADAR-base platform.
These wearables were evaluated in terms of their ease of use, robustness, and the quality of their data collected. To this end, heart rate measurements of 13 participants were taken during different activity phases, using the newly integrated devices and a Fitbit Charge 2 for comparison. Measurements were compared to the Polar H10, as this chest strap uses ECG, which is reported to be a standard in measuring heart rate.

### Getting started

Within [this Jupyter notebook](https://github.com/fschulting/wearables_analysis/blob/main/full_analysis.ipynb) all generated plots and test results are summarized.

In the [sample directory](https://github.com/fschulting/wearables_analysis/blob/main/sample) both sample heart rate data and another Jupyter notebook can be found for test purposes.

### Dependencies

Python 3.10.9 was used in combination with the following libraries:
Pandas 1.5.3, NumPy 1.23.5, SciPy 1.10.0, Matplotlib 3.7.0 and Sklearn 1.2.1.
