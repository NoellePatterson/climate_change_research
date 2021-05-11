import glob
import os
import pandas as pd
import numpy as np
from utils import import_ffc_data, import_ffc_data, import_drh_data, make_results_dicts, summarize_data, summarize_data_no_classes, \
make_summary_dicts, preprocess_dwr, create_model_tables, combine_image, combine_mk_model_stats, gini_index_mk_trends
from trends import calc_mk_trend
from hydrograph import hydrograph, site_hydrograph
from planning_horizons import planning_horizon
# from visualize import plot_drh, plot_rh, line_plots, scatterplot_temp_precip, scatterplot, boxplot, jitterplot

# run with raw flow data from DWR dss files to prepare it for running through the FFC. Files stored in outputs folder. Only run once for new data. 
# data = preprocess_dwr()

model_folders = glob.glob('data_outputs/FFC_results/*85')
ffc_data_all = []
rh_data_all = []
# test = gini_index_mk_trends()
for folder in model_folders:
    # run with FFC outputs (copy and paste from FFC) to combine results files and convert to useable format. Use natural flow class #2 
    ffc_data, model_name = import_ffc_data(folder)
    for data in ffc_data:
        ffc_data_all.append(data)
    drh_data, rh_data = import_drh_data(folder)
    for data in rh_data:
        rh_data_all.append(data)
hydro = site_hydrograph(ffc_data_all, rh_data_all)
#     # Use FFC output files to prepare data for plotting 
#     # drh_data, rh_data = import_drh_data()

#     # Plotting tools using preprocessing outputs from functions above
#     # plots = plot_drh(drh_data)
#     # rh_plot = plot_rh(rh_data)
#     # line_plots = line_plots(ffc_data)
#     # scatter_plot = scatterplot(ffc_data)
#     # result = combine_image()
#     # boxplot = boxplot(ffc_data)
#     # jitterplot = jitterplot(ffc_data)

#     # Generate annotated hydrographs of DWR modeled flow data
#     # planning_horizon_data = planning_horizon(ffc_data)
#     # hydrograph = hydrograph(ffc_data, rh_data)

#     # Statistical analysis tool using preprocessing outputs from above
#     # por_info = make_summary_dicts(ffc_data)
#     results_dicts = make_results_dicts(ffc_data)
#     mk_trend = calc_mk_trend(ffc_data, results_dicts, model_name) 
    
#     # summary = summarize_data_no_classes(results_dicts)
#     # summary = create_model_tables(ffc_data)

