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
    fig, ax = plt.subplots()
    tim_metric = 'FA_Tim'
    mag_metric = 'FA_Mag'
    for model in ffc_data:
        plt_color = 'grey'
        colors_dict = {'1':'mistyrose', '2':'lightcoral', '3':'crimson', '4':'firebrick', '5':'darkred'}
        colors_dict_precip = {'-30':'darkred', '-20':'crimson', '-10':'lightcoral', '10':'dodgerblue', '20':'blue', '30':'darkblue'}
        for key in enumerate(colors_dict_precip):
            # import pdb; pdb.set_trace()
            # if model['gage_id'][8] == key[1]: # for temp-based coloring
            start = model['gage_id'].index('P')
            result = re.findall(r'P([0-9.-]*[0-9]+)', model['gage_id'][start:]) # for precip mag-based coloring
            if result[0] == key[1]: # for precip magnitude-based coloring
                plt_color = colors_dict_precip[key[1]]
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
    plt.title('Fall Pulse')
    ax.legend(loc='upper left')
    plt.show()
    import pdb; pdb.set_trace()
    # In each plot, include ALL values from all models (except control). light shaded points. 
    # Draw over the min/max endpoints, in each axis
    # For each model, determine %exceedance over eco endpoints. (for each metric)
    return