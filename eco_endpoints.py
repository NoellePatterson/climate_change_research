import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import re

def eco_endpoints(ffc_data, rh_data):
    # define the eco endpoints. 5-95th of control? table of endpoints for each ffm
    for model_index, model in enumerate(ffc_data):
        model['ffc_metrics'] = model['ffc_metrics'].apply(pd.to_numeric, errors='coerce')
        if model['gage_id'] == 'SACSMA_CTR_T0P0S0E0I0':
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
    # define hydrograph trace to overlay in plot
    for model_index, model in enumerate(rh_data):
        if model['name'] == 'SACSMA_CTR_T0P0S0E0I0': 
            model['data'] = model['data'].apply(pd.to_numeric, errors='coerce')
            hydrograph_ctrl = model['data']['1979']

    def eco_endpoints_plot(ffc_data, endpoints, hydrograph_ctrl):
        fig, ax = plt.subplots()
        tim_metric = 'DS_Tim' # FA_Tim, SP_Tim, Wet_Tim, 
        mag_metric = 'DS_Mag_50' # FA_Mag, SP_Mag, Wet_BFL_Mag_50, 
        param = 'Seasonal intensity'
        season = 'Dry Season eco-exceedance' # Fall Pulse, Spring Recession, Wet Season, 
        for model in ffc_data:
            # import pdb; pdb.set_trace()
            plt_color = 'grey'
            colors_dict_temp = {'1':'mistyrose', '2':'lightcoral', '3':'crimson', '4':'firebrick', '5':'darkred'}
            colors_dict_precip = {'-30':'darkred', '-20':'crimson', '-10':'lightcoral', '10':'dodgerblue', '20':'blue', '30':'darkblue'}
            colors_dict_int = {'1':'#D9FFBF', '2':'#85CC6F', '3':'#6AB155', '4':'green', '5':'darkgreen'}
            # for key in enumerate(colors_dict_int):
            # import pdb; pdb.set_trace()
            if model['gage_id'].find('OAT') >= 0: # check if it is an OAT model
                if model['gage_id'][10] == 'T':
                    plt_marker = 'o'
                    plt_color_key = model['gage_id'].split('_')[2][1]
                    plt_color = colors_dict_temp[plt_color_key]
                    plt_label = 'temperature'
                elif model['gage_id'][10] == 'P':
                    plt_marker = '^'
                    plt_color_key = re.findall(r'P([0-9.-]*[0-9]+)', model['gage_id'])[0]
                    plt_color = colors_dict_precip[plt_color_key]
                    plt_label = 'precipitation volume'
                elif model['gage_id'][10] == 'S':
                    plt_marker = 'p'
                    plt_color_key = re.findall(r'S([0-9.-]*[0-9]+)', model['gage_id'])[0]
                    plt_color = colors_dict_int[plt_color_key]
                    plt_label = 'seasonal variability'
                elif model['gage_id'][10] == 'E':
                    plt_marker = 'X'
                    plt_color_key = re.findall(r'E([0-9.-]*[0-9]+)', model['gage_id'])[0]
                    plt_color = colors_dict_int[plt_color_key]
                    plt_label = 'event intensity'
                elif model['gage_id'][10] == 'I':
                    plt_marker = 'd'
                    plt_color_key = re.findall(r'I([0-9.-]*[0-9]+)', model['gage_id'])[0]
                    plt_color = colors_dict_int[plt_color_key]
                    plt_label = 'interannual variability'
            elif model['gage_id'].find('EXT') >= 0:
                plt_marker = 'o'
                plt_color = 'black'
                plt_label = 'extreme scenario'
            elif model['gage_id'].find('MID') >= 0:
                plt_marker = 'o'
                plt_color = 'grey'
                plt_label = 'mid-range scenario'
            elif model['gage_id'].find('CTR') >= 0:
                continue

            # import pdb; pdb.set_trace()
            x = model['ffc_metrics'].loc[tim_metric]
            y = model['ffc_metrics'].loc[mag_metric]
            # ax.scatter(x, y, color=plt_color, marker = plt_marker, alpha=0.3, label = plt_label)
            if model['gage_id'] in ('SACSMA_OATT_T5P0S0E0I0', 'SACSMA_OATP_T0P30S0E0I0', 'SACSMA_OATS_T0P0S5E0I0', 'SACSMA_OATE_T0P0S0E5I0',\
                'SACSMA_OATI_T0P0S0E0I5', 'SACSMA_EXT_T0P30S5E5I5', 'SACSMA_MID_T3.4P3.4I1.7'):
                ax.scatter(x, y, color=plt_color, marker = plt_marker, alpha=0.5, label = plt_label)
            else:
                ax.scatter(x, y, color=plt_color, marker = plt_marker, alpha=0.5)
        
        # add min/max endpoints
        plt.vlines(endpoints['eco_5'][tim_metric], ymin=endpoints['eco_5'][mag_metric], ymax=endpoints['eco_95'][mag_metric], color='black')
        plt.vlines(endpoints['eco_95'][tim_metric], ymin=endpoints['eco_5'][mag_metric], ymax=endpoints['eco_95'][mag_metric], color='black')
        plt.hlines(endpoints['eco_5'][mag_metric], xmin=endpoints['eco_5'][tim_metric], xmax=endpoints['eco_95'][tim_metric], label='10-90% control', color='black')
        plt.hlines(endpoints['eco_95'][mag_metric], xmin=endpoints['eco_5'][tim_metric], xmax=endpoints['eco_95'][tim_metric], color='black')

        plt.vlines(endpoints['eco_min'][tim_metric], ymin=endpoints['eco_min'][mag_metric], ymax=endpoints['eco_max'][mag_metric], alpha=0.5, linestyles='dashed', color='black')
        plt.vlines(endpoints['eco_max'][tim_metric], ymin=endpoints['eco_min'][mag_metric], ymax=endpoints['eco_max'][mag_metric], alpha=0.5, linestyles='dashed', color='black')
        plt.hlines(endpoints['eco_min'][mag_metric], xmin=endpoints['eco_min'][tim_metric], xmax=endpoints['eco_max'][tim_metric], label='Full range control', alpha=0.5, linestyles='dashed', color='black')
        plt.hlines(endpoints['eco_max'][mag_metric], xmin=endpoints['eco_min'][tim_metric], xmax=endpoints['eco_max'][tim_metric], alpha=0.5, linestyles='dashed', color='black')
        # add hydrograph trace over top
        plt.plot(hydrograph_ctrl, color='goldenrod', linewidth=2)
        month_ticks = [0,32,60,91,121,152,182,213,244,274,305,335]
        month_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
        plt.xticks(month_ticks, month_labels)
        plt.yticks()
        ax.set_ylabel('Flow (cfs)')
        plt.title(season)
        # ax.legend(loc='upper right') # legend on for Fall Pulse
        # plt.yscale('symlog', linthreshy=10000) # use this for spring and fall plots
        plt.ylim([-200, 9000]) # -200,120000 for fall pulse, -200,170000 for spring, -200, 15000 for wet
        plt.xlim([-10,380]) # -10,380 for fall/spring/wet
        plt.savefig('data_outputs/plots/eco_exceedance/dry.pdf', dpi=1200)
        plt.show()
    
    plots = eco_endpoints_plot(ffc_data, endpoints, hydrograph_ctrl)
    # For each model, determine %exceedance over eco endpoints. (for each metric)
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
    df = pd.DataFrame(data)
    df = df.sort_values('model_name')
    df.to_csv('Eco_endpoints_summary.csv', index=False)
    import pdb; pdb.set_trace()
    # create a dict/table and fill with calc eco exceedance for each metric of model
    # fill exceedances table with appropriate combos of metric exceedances from model-specific dict
    return

def eco_endpoints_slopeplots(ffc_data):
    # assemble results from OAT models
    plot_x = [0, 1, 2, 3, 4, 5, 6]
    temp_dict = {}
    precip_dict = {}
    interann_dict = {}
    seasonal_dict = {}
    event_dict = {}
    dt = ['SACSMA_OATT_T1P0S0E0I0', 'SACSMA_OATT_T2P0S0E0I0', 'SACSMA_OATT_T3P0S0E0I0', 'SACSMA_OATT_T4P0S0E0I0', 'SACSMA_OATT_T5P0S0E0I0']
    dp = ['SACSMA_OATP_T0P-30S0E0I0', 'SACSMA_OATP_T0P-20S0E0I0', 'SACSMA_OATP_T0P-10S0E0I0', 'SACSMA_OATP_T0P10S0E0I0', 'SACSMA_OATP_T0P20S0E0I0', 'SACSMA_OATP_T0P30S0E0I0']
    di = ['SACSMA_OATI_T0P0S0E0I1', 'SACSMA_OATI_T0P0S0E0I2', 'SACSMA_OATI_T0P0S0E0I3', 'SACSMA_OATI_T0P0S0E0I4', 'SACSMA_OATI_T0P0S0E0I5']
    ds = ['SACSMA_OATS_T0P0S1E0I0', 'SACSMA_OATS_T0P0S2E0I0', 'SACSMA_OATS_T0P0S3E0I0', 'SACSMA_OATS_T0P0S4E0I0', 'SACSMA_OATS_T0P0S5E0I0']
    de = ['SACSMA_OATE_T0P0S0E1I0', 'SACSMA_OATE_T0P0S0E2I0', 'SACSMA_OATE_T0P0S0E3I0', 'SACSMA_OATE_T0P0S0E4I0', 'SACSMA_OATE_T0P0S0E5I0']
    for model_index, model in enumerate(ffc_data):
        # import pdb; pdb.set_trace()
        if model['gage_id'] == 'SACSMA_CTR_T0P0S0E0I0':
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
    norm_interann_dict = {}
    norm_seasonal_dict = {}
    norm_event_dict = {}
    norm_precip_dict = {}
    control = control.apply(pd.to_numeric, errors='coerce')
    control = control.mean(axis=1)
    for model in precip_dict.keys():
        # import pdb; pdb.set_trace()
        precip_dict[model] = precip_dict[model].apply(pd.to_numeric, errors='coerce')
        precip_dict[model] = precip_dict[model].mean(axis=1)
    metrics = ffc_data[0]['ffc_metrics'].index
    metrics = metrics.drop(['Peak_5', 'Peak_10', 'Peak_Dur_2', 'Peak_Dur_5', 'Peak_Dur_10', 'Peak_Fre_2', 'Peak_Fre_5', 'Peak_Fre_10', 'Std'])
    for metric in metrics:
        metrics_list = []
        metrics_list.append(control[metric])
        for model in precip_dict.keys():
            metrics_list.append(precip_dict[model][metric])
        norm_min = min(metrics_list)
        norm_max = max(metrics_list)
        for index, val in enumerate(metrics_list):
            val = (val - norm_min)/(norm_max - norm_min)
            metrics_list[index] = val
        norm_precip_dict[metric] = metrics_list
    # make plot with a scatter line for each metric. Label by OAT model.
    # need a way to flip metrics that tend to descend (eg if control is lowest or highest?)
    
    # for slope in norm_precip_dict:
        # if norm_precip_dict[slope][0] > 0.5: # works for temp
        # if norm_precip_dict[slope][0] > 0.5: # works for intensity
        #     norm_precip_dict[slope].reverse()
    fig, ax = plt.subplots()
    for metric in metrics:
        y = norm_precip_dict[metric]
        ax.plot(plot_x, y, alpha=0.3, linestyle='--', marker='o')
    plt.title('Normalized change for event precipitation models')
    plt.show()
    import pdb; pdb.set_trace()

    return()
