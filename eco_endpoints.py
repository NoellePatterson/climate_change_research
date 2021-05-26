import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

def eco_endpoints(ffc_data, rh_data):
    # bring in all the data. 
    # define the eco endpoints. 5-95th of control? table of endpoints for each ffm
    for model_index, model in enumerate(ffc_data):
        model['ffc_metrics'] = model['ffc_metrics'].apply(pd.to_numeric, errors='coerce')
        if model['gage_id'] == 'SACSMA_DT1_DP1_DI0.0':
            control = ffc_data[model_index]['ffc_metrics']
    metrics = ffc_data[0]['ffc_metrics'].index
    eco_min = []
    eco_max = []
    for metric in metrics:
        eco_min.append(np.nanquantile(control.loc[metric], 0.05))
        eco_max.append(np.nanquantile(control.loc[metric], 0.95))
    endpoints = pd.DataFrame(data=[eco_min, eco_max, metrics], index = ['eco_min', 'eco_max', 'metrics'])
    endpoints = endpoints.transpose()
    endpoints = endpoints.set_index(keys='metrics')
    # Start w plots for mag/tim for each FFM. 
    fig, ax = plt.subplots()
    tim_metric = 'FA_Tim'
    mag_metric = 'FA_Mag'
    for model in ffc_data:
        x = model['ffc_metrics'].loc[tim_metric]
        y = model['ffc_metrics'].loc[mag_metric]
        ax.scatter(x, y, color='green', alpha=0.3)
    
    # add min/max endpoints
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