import glob
import os
import pandas as pd
import numpy as np
from utils import import_ffc_data, import_dwr_data, import_drh_data, make_results_dicts, summarize_data, summarize_data_no_classes, make_summary_dicts, preprocess_dwr
from trends import calc_mk_trend
from hydrograph import hydrograph
from planning_horizons import planning_horizon
from visualize import plot_drh, plot_rh, line_plots, scatterplot

# run with raw flow data from DWR dss files to prepare it for running through the FFC. Files stored in outputs folder. Only run once for new data. 
# data = preprocess_dwr()

# run with FFC outputs (copy and paste from FFC) to combine results files and convert to useable format. Use natural flow class #2 
ffc_data = import_dwr_data()

# Use FFC output files to prepare data for plotting 
# drh_data, rh_data = import_drh_data()

# Plotting tools using preprocessing outputs from functions above
# plots = plot_drh(drh_data)
# rh_plot = plot_rh(rh_data)
line_plots = line_plots(ffc_data)
# scatter_plot = scatterplot(ffc_data)

# Generate annotated hydrographs of DWR modeled flow data
planning_horizon_data = planning_horizon(ffc_data)
# hydrograph = hydrograph(ffc_data, rh_data)

# Statistical analysis tool using preprocessing outputs from above
# por_info = make_summary_dicts(ffc_data)
# results_dicts = make_results_dicts(ffc_data)
# mk_trend = calc_mk_trend(ffc_data, results_dicts)
# summary = summarize_data_no_classes(results_dicts)

