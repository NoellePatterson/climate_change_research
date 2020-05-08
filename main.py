import glob
import os
import pandas as pd
import numpy as np
from utils import import_ffc_data, make_results_dicts, summarize_data, make_summary_dicts
from calculations.trends import calc_mk_trend

ffc_data = import_ffc_data()
por_info = make_summary_dicts(ffc_data)
# import pdb; pdb.set_trace()
results_dicts = make_results_dicts(ffc_data)
mk_trend = calc_mk_trend(ffc_data, results_dicts)
summary = summarize_data(results_dicts)