import glob
import pandas as pd
import numpy as np

def import_ffc_data():
    class_folders = glob.glob('data_inputs/ffc_metrics_historic/*')
    ffc_dicts = []
    supp_dicts = []
    gages = []
    for class_folder in class_folders:
        class_name = int(class_folder[-1])
        main_metric_files = glob.glob(class_folder + '/*flow_result.csv')
        supp_metric_files = glob.glob(class_folder + '/*supplementary_metrics.csv')
        for supp_file in supp_metric_files:
            supp_dict = {}
            supp_dict['gage_id'] = int(supp_file[46:54])
            supp_dict['supp_metrics'] = pd.read_csv(supp_file, sep=',', index_col=0)
            supp_dicts.append(supp_dict)
        for metric_file in main_metric_files:
            gage_dict = int(metric_file[46:54])
            # create dictionary for each gage named after gage id, with class and metric results inside 
            gage_dict = {}
            gage_dict['gage_id'] = int(metric_file[46:54])
            gage_dict['class'] = class_name
            for supp_dict in supp_dicts:
                if supp_dict['gage_id'] == gage_dict['gage_id']:
                    # add supp_dict metrics to gage_dict metrics
                    main_metrics = pd.read_csv(metric_file, sep=',', index_col=0)
                    gage_dict['ffc_metrics'] = pd.concat([main_metrics, supp_dict['supp_metrics']], axis=0)
            ffc_dicts.append(gage_dict)
            # Or, store gage info in class instances
            # gage_id = int(metric_file[46:54])
            # ffc_metrics = pd.read_csv(metric_file, sep=',', index_col=None)
            # current_gage = Gage(gage_id, class_name, ffc_metrics)
       
    return ffc_dicts
        
def make_summary_dicts(ffc_data):
    summary_df = pd.DataFrame(columns = ['gage_id', 'class', 'start_yr', 'end_yr', 'POR_len'])
    gage_id = []
    for index, gage_dict in enumerate(ffc_data):
        por_len = len(gage_dict['ffc_metrics'].columns)
        start_yr = int(gage_dict['ffc_metrics'].columns[0])
        end_yr = int(gage_dict['ffc_metrics'].columns[-1])
        summary_df = summary_df.append({'gage_id':gage_dict['gage_id'], 'class':gage_dict['class'], \
        'start_yr':start_yr, 'end_yr':end_yr, 'POR_len':por_len}, ignore_index=True)
    return summary_dicts

def make_results_dicts(ffc_data):
    all_results = []
    metrics_list = ffc_data[0]['ffc_metrics'].index
    for index, gage_dict in enumerate(ffc_data):
        current_gage = {}
        results = pd.DataFrame(index=metrics_list)
        current_gage['gage_id'] = gage_dict['gage_id']
        current_gage['class'] = gage_dict['class']
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


