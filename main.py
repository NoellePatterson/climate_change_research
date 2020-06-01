import glob
import os
import pandas as pd
import numpy as np
from utils import import_ffc_data, import_dwr_data, import_drh_data, make_results_dicts, summarize_data, summarize_data_no_classes, make_summary_dicts, preprocess_dwr
from calculations.trends import calc_mk_trend
from visualize import plot_drh

# data = preprocess_dwr()

# ffc_data = import_dwr_data()
drh_data = import_drh_data()
plots = plot_drh(drh_data)
# por_info = make_summary_dicts(ffc_data)
# results_dicts = make_results_dicts(ffc_data)
# mk_trend = calc_mk_trend(ffc_data, results_dicts)
# summary = summarize_data_no_classes(results_dicts)

