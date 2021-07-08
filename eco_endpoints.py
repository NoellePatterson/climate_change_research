import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import re

def eco_endpoints(ffc_data):
    # bring in all the data. 
    # define the eco endpoints. 5-95th of control? table of endpoints for each ffm
    for model_index, model in enumerate(ffc_data):
        model['ffc_metrics'] = model['ffc_metrics'].apply(pd.to_numeric, errors='coerce')
        if model['gage_id'] == 'SACSMA_T0P0S0E0I0':
            control = ffc_data[model_index]['ffc_metrics']
    metrics = ffc_data[0]['ffc_metrics'].index
    eco_5 = []
    eco_95 = []
    eco_min = []
    eco_max = []
    for metric in metrics:
        eco_5.append(np.nanquantile(control.loc[metric], 0.05))
        eco_95.append(np.nanquantile(control.loc[metric], 0.95))
        eco_min.append(np.nanmin(control.loc[metric]))
        eco_max.append(np.nanmax(control.loc[metric]))
    endpoints = pd.DataFrame(data=[eco_5, eco_95, eco_min, eco_max, metrics], index = ['eco_5', 'eco_95', 'eco_min', 'eco_max', 'metrics'])
    endpoints = endpoints.transpose()
    endpoints = endpoints.set_index(keys='metrics')
    # Start w plots for mag/tim for each FFM. 

    def eco_endpoints_plot(ffc_data, endpoints):
        fig, ax = plt.subplots()
        tim_metric = 'FA_Tim'
        mag_metric = 'FA_Mag'
        param = 'Seasonal intensity'
        for model in ffc_data:
            plt_color = 'grey'
            colors_dict_temp = {'1':'mistyrose', '2':'lightcoral', '3':'crimson', '4':'firebrick', '5':'darkred'}
            colors_dict_precip = {'-30':'darkred', '-20':'crimson', '-10':'lightcoral', '10':'dodgerblue', '20':'blue', '30':'darkblue'}
            colors_dict_int = {'1':'mistyrose', '2':'lightcoral', '3':'crimson', '4':'firebrick', '5':'darkred'}
            for key in enumerate(colors_dict_int):
                # import pdb; pdb.set_trace()
                # if model['gage_id'][8] == key[1]: # for temp-based coloring
                #     plt_color = colors_dict_int[key[1]]
            
                if 'S' in model['gage_id'][7:]: # for intensity-based coloring
                    if re.findall(r'S([0-9.-]*[0-9]+)', model['gage_id'])[0] == key[1]: 
                        plt_color = colors_dict_int[key[1]]

                # start = model['gage_id'].index('P')
                # result = re.findall(r'P([0-9.-]*[0-9]+)', model['gage_id'][start:]) # for precip mag-based coloring
                # if result[0] == key[1]: # for precip magnitude-based coloring
                #     plt_color = colors_dict_precip[key[1]]
            # import pdb; pdb.set_trace()
            x = model['ffc_metrics'].loc[tim_metric]
            y = model['ffc_metrics'].loc[mag_metric]
            ax.scatter(x, y, color=plt_color, alpha=0.3)
        
        # add min/max endpoints
        plt.vlines(endpoints['eco_5'][tim_metric], ymin=endpoints['eco_5'][mag_metric], ymax=endpoints['eco_95'][mag_metric], alpha=0.5, linestyles='dashed')
        plt.vlines(endpoints['eco_95'][tim_metric], ymin=endpoints['eco_5'][mag_metric], ymax=endpoints['eco_95'][mag_metric], alpha=0.5, linestyles='dashed')
        plt.hlines(endpoints['eco_5'][mag_metric], xmin=endpoints['eco_5'][tim_metric], xmax=endpoints['eco_95'][tim_metric], label='Eco 90% threshold', alpha=0.5, linestyles='dashed')
        plt.hlines(endpoints['eco_95'][mag_metric], xmin=endpoints['eco_5'][tim_metric], xmax=endpoints['eco_95'][tim_metric], alpha=0.5, linestyles='dashed')

        plt.vlines(endpoints['eco_min'][tim_metric], ymin=endpoints['eco_min'][mag_metric], ymax=endpoints['eco_max'][mag_metric])
        plt.vlines(endpoints['eco_max'][tim_metric], ymin=endpoints['eco_min'][mag_metric], ymax=endpoints['eco_max'][mag_metric])
        plt.hlines(endpoints['eco_min'][mag_metric], xmin=endpoints['eco_min'][tim_metric], xmax=endpoints['eco_max'][tim_metric], label='Eco threshold')
        plt.hlines(endpoints['eco_max'][mag_metric], xmin=endpoints['eco_min'][tim_metric], xmax=endpoints['eco_max'][tim_metric])
        ax.set_ylabel('Flow (cfs)')
        ax.set_xlabel('Days')
        plt.title('Fall Pulse '+ param + ' coloring')
        ax.legend(loc='upper left')
        plt.show()
    
    # plots = eco_endpoints_plot(ffc_data, endpoints)
    # For each model, determine %exceedance over eco endpoints. (for each metric)
    # exceedances = pd.DataFrame(data=[[], [], [], [], [], [], [], []], index = ['model_name', 'total_exceedance', 'annual_metrics', 'fall_pulse', 'wet_season', \
    # 'peak_flows', 'spring_recesstion', 'dry_season'])
    model_name = []
    total_exceedance = []
    annual_metrics = []
    fall_pulse = []
    wet_season = []
    peak_flows = []
    spring_recession = []
    dry_season = []
    metrics = metrics.drop(['Peak_5', 'Peak_10', 'Peak_Dur_2', 'Peak_Dur_5', 'Peak_Dur_10', 'Peak_Fre_2', 'Peak_Fre_5', 'Peak_Fre_10', 'Std', 'DS_No_Flow'])
    for model_index, model in enumerate(ffc_data):
        # enter model name into table
        model_name.append(model['gage_id'])
        # create a dict/table and fill with control-based eco limits for each metric - done! endpoints. 
        # create a dict/table and fill with calc eco exceedance for each metric of model
        dict = {}
        for metric in metrics: 
            count = 0
            for val in model['ffc_metrics'].loc[metric]:
                if val < endpoints['eco_min'][metric] or val > endpoints['eco_max'][metric]:
                    count += 1
            dict[metric] = count/len(model['ffc_metrics'].loc[metric])
        total_exceedance.append(sum(dict.values()) / len(dict) * 100)
        annual_metrics.append(sum([dict['Avg'], dict['CV']]) / 2 * 100)
        fall_pulse.append(sum([dict['FA_Mag'], dict['FA_Dur'], dict['FA_Tim']]) / 3 * 100)
        wet_season.append(sum([dict['Wet_BFL_Mag_10'], dict['Wet_BFL_Mag_50'], dict['Wet_Tim'], dict['Wet_BFL_Dur']]) / 4 * 100)
        peak_flows.append(dict['Peak_2'] * 100)
        spring_recession.append(sum([dict['SP_Mag'], dict['SP_Tim'], dict['SP_Dur'], dict['SP_ROC']]) / 4 * 100)
        dry_season.append(sum([dict['DS_Mag_50'], dict['DS_Mag_90'], dict['DS_Tim'], dict['DS_Dur_WS']]) / 4 * 100)
    data = {'model_name':model_name, 'total_exceedance':total_exceedance, 'annual_metrics':annual_metrics, 'fall_pulse':fall_pulse, \
        'wet_season':wet_season, 'peak_flows':peak_flows, 'spring_recession':spring_recession, 'dry_season':dry_season}
    # df = pd.DataFrame([total_exceedance, annual_metrics, fall_pulse, wet_season, peak_flows, spring_recession, dry_season], index=model_name), 
    # columns=['total_exceedance', 'annual_metrics', 'fall_pulse', 'wet_season', 'peak_flows', 'spring_recession', 'dry_season'])
    df = pd.DataFrame(data)
    df = df.sort_values('model_name')
    df.to_csv('Eco_endpoints_summary.csv', index=False)
    import pdb; pdb.set_trace()
    # create a dict/table and fill with calc eco exceedance for each metric of model
    # fill exceedances table with appropriate combos of metric exceedances from model-specific dict
    return

def eco_endpoints_slopeplots(ffc_data):
    # assemble results from OAT models
    plot_x = [0, 1, 2, 3, 4, 5]
    temp_dict = {}
    precip_dict = {}
    interann_dict = {}
    seasonal_dict = {}
    event_dict = {}
    dt = ['SACSMA_T1P0S0E0I0', 'SACSMA_T2P0S0E0I0', 'SACSMA_T3P0S0E0I0', 'SACSMA_T4P0S0E0I0', 'SACSMA_T5P0S0E0I0']
    dp = ['SACSMA_T0P-30S0E0I0', 'SACSMA_T0P-20S0E0I0', 'SACSMA_T0P-10S0E0I0', 'SACSMA_T0P10S0E0I0', 'SACSMA_T0P20S0E0I0', 'SACSMA_T0P30S0E0I0']
    di = ['SACSMA_T0P0S0E0I1', 'SACSMA_T0P0S0E0I2', 'SACSMA_T0P0S0E0I3', 'SACSMA_T0P0S0E0I4', 'SACSMA_T0P0S0E0I5']
    ds = ['SACSMA_T0P0S1E0I0', 'SACSMA_T0P0S2E0I0', 'SACSMA_T0P0S3E0I0', 'SACSMA_T0P0S4E0I0', 'SACSMA_T0P0S5E0I0']
    de = ['SACSMA_T0P0S0E1I0', 'SACSMA_T0P0S0E2I0', 'SACSMA_T0P0S0E3I0', 'SACSMA_T0P0S0E4I0', 'SACSMA_T0P0S0E5I0']
    for model_index, model in enumerate(ffc_data):
        # import pdb; pdb.set_trace()
        if model['gage_id'] == 'SACSMA_T0P0S0E0I0':
            control = ffc_data[model_index]['ffc_metrics']
        for temp_model in dt:
            if model['gage_id'] == temp_model:
                temp_dict[temp_model] = ffc_data[model_index]['ffc_metrics']
        for precip_model in dp:
            if model['gage_id'] == precip_model:
                precip_dict[precip_model] = ffc_data[model_index]['ffc_metrics']
        for interann_model in di:
            if model['gage_id'] == interann_model:
                interann_dict[interann_model] = ffc_data[model_index]['ffc_metrics']
        for seasonal_model in ds:
            if model['gage_id'] == seasonal_model:
                seasonal_dict[seasonal_model] = ffc_data[model_index]['ffc_metrics']
        for event_model in de:
            if model['gage_id'] == event_model:
                event_dict[event_model] = ffc_data[model_index]['ffc_metrics']

    # normalize each set of 6 metrics across all models (min is 0, max is 1)
    # start with DT 
    # create list with vals - mean ffc metric val for each model
    # normalize vals in those lists (of ffc metrics)
    norm_temp_dict = {}
    control = control.apply(pd.to_numeric, errors='coerce')
    control = control.mean(axis=1)
    for model in temp_dict.keys():
        # import pdb; pdb.set_trace()
        temp_dict[model] = temp_dict[model].apply(pd.to_numeric, errors='coerce')
        temp_dict[model] = temp_dict[model].mean(axis=1)
    metrics = ffc_data[0]['ffc_metrics'].index
    metrics = metrics.drop(['Peak_5', 'Peak_10', 'Peak_Dur_2', 'Peak_Dur_5', 'Peak_Dur_10', 'Peak_Fre_2', 'Peak_Fre_5', 'Peak_Fre_10', 'Std'])
    for metric in metrics:
        temp_list = []
        temp_list.append(control[metric])
        for model in temp_dict.keys():
            temp_list.append(temp_dict[model][metric])
        norm_min = min(temp_list)
        norm_max = max(temp_list)
        for index, val in enumerate(temp_list):
            val = (val - norm_min)/(norm_max - norm_min)
            temp_list[index] = val
        norm_temp_dict[metric] = temp_list
    # make plot with a scatter line for each metric. Label by OAT model.
    # need a way to flip metrics that tend to descend (eg if control is lowest or highest?)
    
    for slope in norm_temp_dict:
        if norm_temp_dict[slope][0] > 0.5:
            norm_temp_dict[slope].reverse()
    fig, ax = plt.subplots()
    for metric in metrics:
        y = norm_temp_dict[metric]
        ax.plot(plot_x, y, alpha=0.3, linestyle='--', marker='o')
    plt.show()
    import pdb; pdb.set_trace()

 

    return()
