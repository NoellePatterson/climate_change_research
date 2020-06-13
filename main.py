import glob
import os
import pandas as pd
import numpy as np
from utils import import_ffc_data, import_dwr_data, import_drh_data, make_results_dicts, summarize_data, summarize_data_no_classes, make_summary_dicts, preprocess_dwr
from calculations.trends import calc_mk_trend
from visualize import plot_drh, plot_rh, line_plots, scatterplot

# data = preprocess_dwr()

ffc_data = import_dwr_data()
drh_data, rh_data = import_drh_data()
# plots = plot_drh(drh_data)
rh_plot = plot_rh(rh_data)
# line_plots = line_plots(ffc_data)
# por_info = make_summary_dicts(ffc_data)
# results_dicts = make_results_dicts(ffc_data)
# mk_trend = calc_mk_trend(ffc_data, results_dicts)
# summary = summarize_data_no_classes(results_dicts)

