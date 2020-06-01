import glob
import pandas as pd
import numpy as np
from datetime import timedelta

def import_ffc_data():
    class_folders = glob.glob('data_inputs/ffc_metrics_historic/*')
    ffc_dicts = []
    supp_dicts = []
    gages = []
    for class_folder in class_folders:
        class_name = int(class_folder[-1])
        main_metric_files = glob.glob(class_folder + '/*flow_result.csv')
        supp_metric_files = glob.glob(class_folder + '/*supplementary_metrics.csv')
        # pull out supplementary metric information for each gage
        for supp_file in supp_metric_files:
            supp_dict = {}
            supp_dict['gage_id'] = int(supp_file[46:54])
            supp_dict['supp_metrics'] = pd.read_csv(supp_file, sep=',', index_col=0)
            supp_dicts.append(supp_dict)
        for metric_file in main_metric_files:
            main_metrics = pd.read_csv(metric_file, sep=',', index_col=0)
            # optional filter out gages by POR length and end date
            por_len = len(main_metrics.columns)
            por_end = int(main_metrics.columns[-1])
            if por_len < 40 or por_end < 2005:
                continue 

            gage_dict = int(metric_file[46:54])
            # create dictionary for each gage named after gage id, with class and metric results inside 
            gage_dict = {}
            gage_dict['gage_id'] = int(metric_file[46:54])
            gage_dict['class'] = class_name
            # align supplemental metric file with main metric file, and add info to the main gage dict
            for supp_dict in supp_dicts:
                if supp_dict['gage_id'] == gage_dict['gage_id']:
                    # add supp_dict metrics to gage_dict metrics
                    gage_dict['ffc_metrics'] = pd.concat([main_metrics, supp_dict['supp_metrics']], axis=0)
            ffc_dicts.append(gage_dict) 
    return ffc_dicts

def import_dwr_data():
    main_metric_files = glob.glob('data_inputs/dwr_ffc_results' + '/*flow_result.csv')
    supp_metric_files = glob.glob('data_inputs/dwr_ffc_results' + '/*supplementary_metrics.csv')
    ffc_dicts = []
    supp_dicts = []
    for supp_file in supp_metric_files:
        supp_dict = {}
        supp_dict['gage_id'] = supp_file.split('_')[3].split('/')[1]
        supp_dict['supp_metrics'] = pd.read_csv(supp_file, sep=',', index_col=0)
        supp_dicts.append(supp_dict)
    for metric_file in main_metric_files:
        main_metrics = pd.read_csv(metric_file, sep=',', index_col=0)
        # create dictionary for each gage named after gage id, with class and metric results inside 
        gage_dict = {}
        gage_dict['gage_id'] = metric_file.split('_')[3].split('/')[1]
        # align supplemental metric file with main metric file, and add info to the main gage dict
        for supp_dict in supp_dicts:
            if supp_dict['gage_id'] == gage_dict['gage_id']:
                # add supp_dict metrics to gage_dict metrics
                gage_dict['ffc_metrics'] = pd.concat([main_metrics, supp_dict['supp_metrics']], axis=0)
        ffc_dicts.append(gage_dict)
    return ffc_dicts

def import_drh_data():
    drh_files = glob.glob('data_inputs/dwr_ffc_results' + '/*drh.csv')
    supp_metric_files = glob.glob('data_inputs/dwr_ffc_results' + '/*supplementary_metrics.csv')
    drh_dicts = []
    for index, drh_file in enumerate(drh_files):
        drh_dict = {}
        drh_dict['name'] = drh_file.split('_')[3].split('/')[1]
        drh_dict['data'] = pd.read_csv(drh_file, sep=',', index_col=0, header=None)
        drh_dicts.append(drh_dict)
    return drh_dicts
        
def make_summary_dicts(ffc_data):
    summary_df = pd.DataFrame(columns = ['gage_id', 'class', 'start_yr', 'end_yr', 'POR_len'])
    gage_id = []
    for index, gage_dict in enumerate(ffc_data):
        por_len = len(gage_dict['ffc_metrics'].columns)
        start_yr = int(gage_dict['ffc_metrics'].columns[0])
        end_yr = int(gage_dict['ffc_metrics'].columns[-1])
        summary_df = summary_df.append({'gage_id':gage_dict['gage_id'], 'class':gage_dict['class'], \
        'start_yr':start_yr, 'end_yr':end_yr, 'POR_len':por_len}, ignore_index=True)
    summary_df.to_csv('data_outputs/por.csv')
    return summary_df

def make_results_dicts(ffc_data):
    all_results = []
    metrics_list = ffc_data[0]['ffc_metrics'].index
    for index, gage_dict in enumerate(ffc_data):
        current_gage = {}
        results = pd.DataFrame(index=metrics_list)
        current_gage['gage_id'] = gage_dict['gage_id']
        # current_gage['class'] = gage_dict['class'] # use with FFC reference data, which has class
        current_gage['results'] = results
        all_results.append(current_gage)
    return all_results

def summarize_data(results_dicts):
    classes = [[] for i in range(9)]
    for gage_dict in results_dicts:
        for i in range(1,10):
            if gage_dict['class'] == i:
                # have to align indexing of classes starting at 1 with Python's default 0-indexing
                classes[i-1].append(gage_dict)
    metrics = results_dicts[0]['results'].index
    summary_df = pd.DataFrame(index=metrics)
    # look in each class for that metric
    for index in range(1,10):
        current_class = classes[index-1]
        summary_df['class_{}_down'.format(index)] = np.nan
        summary_df['class_{}_no_trend'.format(index)] = np.nan
        summary_df['class_{}_up'.format(index)] = np.nan
        for metric in metrics: 
            down_trends = 0
            no_trends = 0
            up_trends = 0
            for gage in current_class:
                mk_decision = gage['results'].loc[metric,'mk_decision']
                if mk_decision == 'decreasing':
                    down_trends += 1
                elif mk_decision == 'no trend':
                    no_trends += 1
                elif mk_decision == 'increasing':
                    up_trends += 1
            summary_df.loc[metric, 'class_{}_down'.format(index)] = down_trends
            summary_df.loc[metric, 'class_{}_no_trend'.format(index)] = no_trends
            summary_df.loc[metric, 'class_{}_up'.format(index)] = up_trends
    summary_df.to_csv('data_outputs/mk_summary.csv')

def summarize_data_no_classes(results_dicts):
    metrics = results_dicts[0]['results'].index
    summary_df = pd.DataFrame(index=metrics)
    summary_df['Down'] = np.nan
    summary_df['No_trend'] = np.nan
    summary_df['Up'] = np.nan
    for metric in metrics: 
        down_trends = 0
        no_trends = 0
        up_trends = 0
        for gage in results_dicts:
            mk_decision = gage['results'].loc[metric,'mk_decision']
            if mk_decision == 'decreasing':
                down_trends += 1
            elif mk_decision == 'no trend':
                no_trends += 1
            elif mk_decision == 'increasing':
                up_trends += 1
        summary_df.loc[metric, 'Down'] = down_trends
        summary_df.loc[metric, 'No_trend'] = no_trends
        summary_df.loc[metric, 'Up'] = up_trends
    summary_df.to_csv('data_outputs/mk_summary_dwr.csv')


def preprocess_dwr():
    files = glob.glob('data_inputs/DWR_data/*')
    for file in files:
        df = pd.read_csv(file, names = ['date','flow'], parse_dates=['date'])  
        # Dates were erroneously set a century late in all data from 1969 and earlier, so need to correct
        for index, value in enumerate(df['date']):
        # find index of first date to hit 1970
            if pd.to_datetime(value, format='%m/%d/%Y') == pd.to_datetime('01011970', format='%m%d%Y'):
                change_century_index = index
                break
        # all rows from start of data until 1970 get 100 years removed from date
        for index, value in enumerate(df['date'][0:index]):
            df.loc[index, 'date'] -= timedelta(days=365.24*100)
        df['date'] = df['date'].dt.strftime('%m/%d/%Y')
        df.to_csv('data_outputs/'+file, index=False)
