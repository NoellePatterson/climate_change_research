# -*- coding: utf-8 -*-
"""
Collect and process gridded runoff data from Scripps
Noelle Patterson, UC Davis, 2020
"""
import xarray as xr
import glob
from matplotlib import pyplot as plt
import pandas as pd
import os
import re
import numpy as np
import csv
from datetime import date, timedelta

# #open up nc files
# nc_files = glob.glob('data_inputs/runoff_grids/*.nc')
# for nc_data in nc_files:
#     dataset = xr.open_dataset(nc_data)
#     runoff = dataset['runoff']
#     year = int(nc_data[20:24])
#     import pdb; pdb.set_trace()
#     if not os.path.exists('data_outputs/{}'.format(year)):
#         os.makedirs('data_outputs/{}'.format(year))
#     # loop through all values in the (195)
#     for index in range(len(runoff[0])):
#         timeseries = []
#         # loop through all 365 days
#         for i in range(len(runoff)):
#             timeseries.append(float(runoff[i][index][10]))
#         df = pd.DataFrame(timeseries) 
#         df.fillna(0)
#         df.to_csv('data_outputs/{}/timeseries_{}.csv'.format(year, index), index=False, header=False)
#     print('year {} done!'.format(year))

# Stitch 1-year timeseries together into 100-year timeseries
# files = {}
# for i in range(2006, 2101, 1):
#     dirs = glob.glob('data_outputs/timeseries_raw/{}/*.csv'.format(i))
#     files[str(i)] = sorted(dirs, key=lambda x: float(re.findall(r'-?\d+\.?\d*', x)[1]))
# for site in range(195):
#     #0
#     timeseries = []
#     # arrange glob files into numeric order to read in 1 @ a time
#     for year in range(2006, 2101, 1):
#         # import pdb; pdb.set_trace()
#         data = pd.read_csv(files[str(year)][site]).iloc[:,0].values.tolist()
#         timeseries.append(data)
#     for index, value in enumerate(timeseries):
        
#         np.array(timeseries[index])[np.isnan(value)] = 0
#     flat_timeseries = [item for sublist in timeseries for item in sublist]
#     # flat_timeseries = []
#     # for sublist in timeseries:
#     #     for item in sublist:
#     #         flat_timeseries.append(item)
    
#     dates = pd.date_range('01/01/2006', periods=len(flat_timeseries))
#     df = pd.DataFrame(list(zip(dates, flat_timeseries)), columns = ['date', 'flow'])
#     df.to_csv('data_outputs/timeseries/runoff_100_year_site_{}.csv'.format(site), index=False)
#     print('site {} done!'.format(site))

# Add date and flow columns to FFC data, foldered by class

folders = glob.glob('data_inputs/FFC_update_POR/*')
for folder in folders:
    files = glob.glob(folder+'/*.csv')
    for file in files:
        print(file)
        df = pd.read_csv(file, names = ['date','flow'], parse_dates=['date'])
        future_dates = df['date'] > pd.to_datetime('20210101', format='%Y%m%d')
        df.loc[future_dates, 'date'] -= timedelta(days=365.25*100)
        df['date'] = df['date'].dt.strftime('%m/%d/%Y')
        df.dropna(subset = ['date'], inplace=True)
        df.to_csv('data_outputs/'+file, index=False)

