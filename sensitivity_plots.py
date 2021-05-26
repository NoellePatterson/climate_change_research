import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

def sens_plots(ffc_data, rh_data):
    # Averge each FFC metric across the POR.
    for model in ffc_data:
        model['ffc_metrics'] = model['ffc_metrics'].apply(pd.to_numeric, errors='coerce')
        model['ffc_metrics'] = model['ffc_metrics'].mean(axis=1)
    
    # Normalize metric values across all models. 
    ffc_data_norm = copy.deepcopy(ffc_data)
    metrics = ffc_data[0]['ffc_metrics'].index
    for metric in metrics:
        max_metric = ffc_data[0]['ffc_metrics'][metric] # can pick any starting val in dataset and test all others against it for min/max
        min_metric = ffc_data[0]['ffc_metrics'][metric] 
        for model in ffc_data_norm:
            if model['ffc_metrics'][metric] is None:
                continue
            elif model['ffc_metrics'][metric] > max_metric:
                max_metric = model['ffc_metrics'][metric]
            elif model['ffc_metrics'][metric] < min_metric:
                min_metric = model['ffc_metrics'][metric]
        
        for index, model in enumerate(ffc_data_norm):
            ffc_data_norm[index]['ffc_metrics'][metric] = (model['ffc_metrics'][metric] - min_metric)/(max_metric - min_metric)
    # Group together all values for each metric (across models)
    dT_min = []
    dT_mid = []
    dT_min_names = ['SACSMA_DT1_DP1_DI0.0', 'SACSMA_DT2_DP1_DI0.0', 'SACSMA_DT3_DP1_DI0.0', 'SACSMA_DT4_DP1_DI0.0', 'SACSMA_DT5_DP1_DI0.0']
    dT_mid_names = ['SACSMA_DT1_DP1.1_DI0.6', 'SACSMA_DT2_DP1.1_DI0.6', 'SACSMA_DT3_DP1.1_DI0.6', 'SACSMA_DT4_DP1.1_DI0.6', 'SACSMA_DT5_DP1.1_DI0.6']
    for model_index, model in enumerate(ffc_data_norm):
        if model['gage_id'] in dT_min_names:
            dT_min.append(ffc_data[model_index])
        elif model['gage_id'] in dT_mid_names:
            dT_mid.append(ffc_data[model_index])
    
    # Plot: a line for each ffc metric, low PI dT (20 lines, 6 pts each)
    fig, ax = plt.subplots()
    # start with low PI dT (all blue lines)
    # for each metric, form a line from the six models
    for metric in metrics:
        line = []
        for model in dT_min:
            line.append(model['ffc_metrics'][metric])
        x = range(len(line))
        ax.scatter(x, line, color='orange', alpha=0.5)
        z = np.polyfit(x, line, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x),'-', color='orange')

        line_mid = []
        for model in dT_mid:
            line_mid.append(model['ffc_metrics'][metric])
        x = range(len(line_mid))
        ax.scatter(x, line_mid, color='blue', alpha=0.5)
        z = np.polyfit(x, line_mid, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x),'-', color='blue')
    # plt.show()

    # Create summary table of extreme ends of sensitivity analysis
    sens_summary = {}
    # sens_summary_names = ['SACSMA_DT0_DP1_DI0.0', 'SACSMA_DT5_DP1_DI0.0', 'SACSMA_DT0_DP1.3_DI0.0', 'SACSMA_DT0_DP1_DI1.0'] # models at far ends of sensitivity
    sens_summary_names = ['SACSMA_DT0_DP1_DI0.0', 'SACSMA_DT1_DP0.8_DI0.2', 'SACSMA_DT2_DP0.9_DI0.4', 'SACSMA_DT3_DP1.1_DI0.6', 'SACSMA_DT4_DP1.2_DI0.8', 
    'SACSMA_DT5_DP1.3_DI1.0'] # step-wise combination models
    for model_index, model in enumerate(ffc_data):
        if model['gage_id'] in sens_summary_names:
            sens_summary[model['gage_id']] = model['ffc_metrics']
    df = pd.DataFrame(sens_summary)
    df.to_csv('data_outputs/sensitivity_summary_combo_mods.csv')
    import pdb; pdb.set_trace()
    # same plot for mid PI dT
    return 
    