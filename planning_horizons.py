# -*- coding: utf-8 -*-
"""
Use DWR planning horizons to aggregate 30 climate change simulations of ffc metrics into a single timeseries 
of ffc metrics based on the likelihood of each simulation's occurrence in future conditions
Noelle Patterson, UC Davis, 2020
"""
import numpy as np
import pandas as pd


def planning_horizon(ffc_data):
    planning_probs = pd.read_csv('data_inputs/planning_probs.csv', sep=',')
    probs = ['FUT(2026-2055)', 'FUT(2056-2085)']
    # Make empty df to fill with aggregated ffc metrics
    empty_df = ffc_data[0]['ffc_metrics'].copy(deep=True)
    for col in empty_df.columns:
        empty_df[col].values[:] = 0
    import pdb; pdb.set_trace()
    new_dict = {'gage_id':'aggregate','ffc_metrics':empty_df}
    metrics = new_dict['ffc_metrics'].index.tolist()
    # map simulation names to the order presented 
    simulation_map = {'DT0P0.8':0, 'DT0DP0.9':1, 'DT0DP1':2}
    for probability_range in probs:
        planning_probs_current = planning_probs[planning_probs['period']==probability_range]
        for metric in metrics:
            val = 0
            for sim_dict in ffc_data:
                import pdb; pdb.set_trace()
                sim_name = sim_dict['gage_id']
                prob_number = simulation_map[sim_name]
                metric_data = pd.to_numeric(sim_dict['ffc_metrics'].loc[metric], errors='coerce')
                current_prob = planning_probs_current['Biv_Norm_Prob'][prob_number]
                val += metric_data * current_prob
        
        import pdb; pdb.set_trace()
