import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Plot hydrographs with FFC metric overlays for DWR data
'''

def hydrograph(ffc_data, rh_data):
    # Find the control results from the list of all DWR results
    for index, sim in enumerate(ffc_data):
        if sim['gage_id'] == 'DT0DP1':
            control_index = index
    control = ffc_data[control_index]['ffc_metrics']
    control_transpose = control.transpose()
    control_transpose = control_transpose.apply(pd.to_numeric, errors='coerce') # convert all columns to numeric 

    # 1. stratify water year results by WYT
    avg_ann_q = control.loc['Avg']
    dry_thresh = np.percentile(avg_ann_q, 33.333) 
    wet_thresh = np.percentile(avg_ann_q, 66.666)
    dry_q = control_transpose[avg_ann_q <= dry_thresh]
    mod_q = control_transpose[(avg_ann_q > dry_thresh) & (avg_ann_q < wet_thresh)]
    wet_q = control_transpose[avg_ann_q >= wet_thresh]

    # 2. Gather average val FFC metrics across WYT groups
    
    ffc_vals = pd.DataFrame(columns = ['all_years', 'dry', 'mod', 'wet'])
    metrics = ['FA_Tim', 'FA_Mag', 'FA_Dur', 'Wet_Tim', 'Wet_BFL_Mag_50', 'Peak_5', 'Peak_2','Peak_Dur_5','Peak_Dur_2','SP_Tim', 'SP_Mag', 'SP_ROC', 'DS_Tim', 'DS_Mag_50']
    for metric in metrics:
        ffc_vals.loc[metric+'_med', 'all_years'] = control_transpose[metric].median() 
        ffc_vals.loc[metric+'_med', 'dry'] = dry_q[metric].median() 
        ffc_vals.loc[metric+'_med', 'mod'] = mod_q[metric].median()  
        ffc_vals.loc[metric+'_med', 'wet'] = wet_q[metric].median()
    for metric in metrics:
        ffc_vals.loc[metric+'_25', 'all_years'] = control_transpose[metric].quantile(.25) 
        ffc_vals.loc[metric+'_25', 'dry'] = dry_q[metric].quantile(.25) 
        ffc_vals.loc[metric+'_25', 'mod'] = mod_q[metric].quantile(.25) 
        ffc_vals.loc[metric+'_25', 'wet'] = wet_q[metric].quantile(.25)
    for metric in metrics:
        ffc_vals.loc[metric+'_75', 'all_years'] = control_transpose[metric].quantile(.75) 
        ffc_vals.loc[metric+'_75', 'dry'] = dry_q[metric].quantile(.75) 
        ffc_vals.loc[metric+'_75', 'mod'] = mod_q[metric].quantile(.75) 
        ffc_vals.loc[metric+'_75', 'wet'] = wet_q[metric].quantile(.75)

    # 3. plot out hydrograph with 25/75 lines

    # Find the control plotting data from the list of simulations
    for index, sim in enumerate(rh_data):
        if sim['name'] == 'DT0DP1':
            control_index = index
    
    plot_data = rh_data[control_index]
    percentiles = [10, 25, 50, 75, 90]
    percentile_keys = ['ten', 'twenty_five', 'fifty', 'seventy_five', 'ninety']

    rh = {}
    for index, percentile in enumerate(percentile_keys):
        rh[percentile] = []
    for row_index, _ in enumerate(plot_data['data'].iloc[:,0]): # loop through each row, 366 total
        # loop through all five percentiles
        for index, percentile in enumerate(percentiles): 
            # calc flow percentiles across all years for each row of flow matrix
            flow_row = pd.to_numeric(plot_data['data'].iloc[row_index, :], errors='coerce')
            rh[percentile_keys[index]].append(np.nanpercentile(flow_row, percentile))
        
    plt.rc('ytick', labelsize=5) 
    plt.subplot()
    name = plot_data['name']
    # make plot

    x = np.arange(0,366,1)
    ax = plt.plot(rh['ten'], color = 'darkgrey', label = "10%") # original color is navy
    # plt.plot(rh['twenty_five'], color = 'blue', label = "25%")
    plt.plot(rh['fifty'], color='red', label = "50%")
    # plt.plot(rh['seventy_five'], color = 'blue', label = "75%")
    plt.plot(rh['ninety'], color = 'darkgrey', label = "90%") # original color is navy
    # plt.fill_between(x, rh['ten'], rh['twenty_five'], color='powderblue')
    # plt.fill_between(x, rh['twenty_five'], rh['fifty'], color='powderblue')
    # plt.fill_between(x, rh['fifty'], rh['seventy_five'], color='powderblue')
    # plt.fill_between(x, rh['seventy_five'], rh['ninety'], color='powderblue')
    plt.fill_between(x, rh['ten'], rh['fifty'], color='lightgrey')
    plt.fill_between(x, rh['fifty'], rh['ninety'], color='lightgrey')

    plt.title('Control - {} - Dry flow years'.format(name), size=9)
    plt.ylabel('Flow (cfs)')
    tick_spacing = [0, 30.5, 61, 91.5, 122, 152.5, 183, 213.5, 244, 274.5, 305, 335.5]
    tick_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    plt.xticks(tick_spacing, tick_labels)
    plt.tick_params(axis='y', which='major', pad=1)
    plt.axhline(y=0, ls=':', color='darkgrey')
    

    # 4. Plot out boxes based on FFC metric vals. 
    year_type = ffc_vals['dry']

    # Fall pulse component box
    # fall_x = [year_type.loc['FA_Tim_25'], year_type.loc['FA_Tim_75'], year_type.loc['FA_Tim_75'], year_type.loc['FA_Tim_25']]
    # fall_y = [year_type.loc['FA_Mag_75'], year_type.loc['FA_Mag_75'], year_type.loc['FA_Mag_25'], year_type.loc['FA_Mag_25'],]
    # plt.fill(fall_x, fall_y)

    # Wet baseflow component box
    # wet_x = [year_type.loc['Wet_Tim_25'], year_type.loc['Wet_Tim_75'], year_type.loc['Wet_Tim_75'], year_type.loc['Wet_Tim_25']]
    # wet_y = [year_type.loc['Wet_BFL_Mag_50_75'], year_type.loc['Wet_BFL_Mag_50_75'], year_type.loc['Wet_BFL_Mag_50_25'], year_type.loc['Wet_BFL_Mag_50_25'],]
    # plt.fill(wet_x, wet_y)

    # Spring Recession component box
    # spring_x = [year_type.loc['SP_Tim_25'], year_type.loc['SP_Tim_75'], year_type.loc['SP_Tim_75'], year_type.loc['SP_Tim_25']]
    # spring_y = [year_type.loc['SP_Mag_75'], year_type.loc['SP_Mag_75'], year_type.loc['SP_Mag_25'], year_type.loc['SP_Mag_25'],]
    # plt.fill(spring_x, spring_y)

    # Dry season component box
    # dry_x = [year_type.loc['DS_Tim_25'], year_type.loc['DS_Tim_75'], year_type.loc['DS_Tim_75'], year_type.loc['DS_Tim_25']]
    # dry_y = [year_type.loc['DS_Mag_50_75'], year_type.loc['DS_Mag_50_75'], year_type.loc['DS_Mag_50_25'], year_type.loc['DS_Mag_50_25'],]
    # plt.fill(dry_x, dry_y)
    # import pdb; pdb.set_trace()

    '''
    Plot points to form FFC hydrograph
    '''
    # import pdb; pdb.set_trace()
    famed_x = year_type.loc['FA_Tim_med']
    fadur_x = year_type.loc['FA_Dur_med']
    wsmed_x = year_type.loc['Wet_Tim_med']
    # hardcode peak duration value. otherwise won't have any values for dry years
    peakdur_x = ffc_vals['all_years'].loc['Peak_Dur_5_med']
    pre_spmed_x = year_type.loc['SP_Tim_med']-60
    peakmed_x = wsmed_x + (pre_spmed_x - wsmed_x)/2 # pick somewhat arbitary spot for a peak flow vis
    spmed_x = year_type.loc['SP_Tim_med']
    sp_rocmed_x = year_type.loc['SP_Tim_med']+30
    dsmed_x = year_type.loc['DS_Tim_med']
    x_med = [0, famed_x-.5*fadur_x,famed_x,famed_x+.5*fadur_x, wsmed_x,wsmed_x, peakmed_x-peakdur_x/2,peakmed_x,peakmed_x+peakdur_x/2, pre_spmed_x, spmed_x, sp_rocmed_x, dsmed_x, 366]

    famed_y = year_type.loc['FA_Mag_med']
    dsmed_y = year_type.loc['DS_Mag_50_med']
    wsmed_y = year_type.loc['Wet_BFL_Mag_50_med']
    # change peak value to lowest percentile for dry years
    peakmed_y = year_type.loc['Peak_2_med']
    spmed_y = year_type.loc['SP_Mag_med']
    sp_rocmed_y = year_type.loc['SP_Mag_med']-30*year_type.loc['SP_ROC_med']*year_type.loc['SP_Mag_med']
    y_med = [dsmed_y, dsmed_y,famed_y,dsmed_y, dsmed_y,wsmed_y, wsmed_y,peakmed_y,wsmed_y, wsmed_y, spmed_y, sp_rocmed_y, dsmed_y, dsmed_y]
    plt.plot(x_med, y_med, 'b--')

    '''
    Plot 25/57 FFC hydrograph as filled in error bounds
    '''
    # all x values corresponding with both 25th and 75th percentile hydrographs

    ws75_x = year_type.loc['Wet_Tim_75']
    ws25_x = year_type.loc['Wet_Tim_25']
    pre_sp_75_x = year_type.loc['SP_Tim_75']-60
    pre_sp_25_x = year_type.loc['SP_Tim_25']-60
    sp75_x = year_type.loc['SP_Tim_75']
    sp25_x = year_type.loc['SP_Tim_25']
    sp_roc25_x = year_type.loc['SP_Tim_25']+30
    sp_roc75_x = year_type.loc['SP_Tim_75']+30
    ds25_x = year_type.loc['DS_Tim_25']
    ds75_x = year_type.loc['DS_Tim_75']
    # x_vals_fill = [0, ws75, ws75, ws25, ws25, pre_sp_75, pre_sp_25, sp75, sp25, sp_roc25, sp_roc75, ds25, ds75, 366]
    x_25 = [0, ws25_x, ws25_x, pre_sp_25_x, sp25_x, sp_roc25_x, ds25_x]
    x_75 = [0, ws75_x, ws75_x, pre_sp_75_x, sp75_x, sp_roc75_x, ds75_x]

    # all 25th percentile fill y vals
    fa25_y = year_type.loc['FA_Mag_25']
    ds25_y = year_type.loc['DS_Mag_50_25']
    ws25_y = year_type.loc['Wet_BFL_Mag_50_25']
    sp25_y = year_type.loc['SP_Mag_25']
    sp_roc25_y = sp_rocmed_y - (spmed_y - sp25_y)
    # if end of spring recession drops too low based on original magnitude settings, reassign magnitude to dry season value
    if sp_roc25_y < ds25_y:
        sp_roc25_y = ds25_y

    fa75_y = year_type.loc['FA_Mag_75']
    ds75_y = year_type.loc['DS_Mag_50_75']
    ws75_y = year_type.loc['Wet_BFL_Mag_50_75']
    sp75_y = year_type.loc['SP_Mag_75']
    sp_roc75_y = sp_rocmed_y + (sp75_y - spmed_y)
    
    y_25 = [ds25_y, ds25_y, ws25_y, ws25_y, sp25_y, sp_roc25_y, ds25_y]
    y_75 = [ds75_y, ds75_y, ws75_y, ws75_y, sp75_y, sp_roc75_y, ds75_y]

    '''
    Use fill between function to plot 25/75 percentile bounds
    '''
    # fill from start of year to start of wet season, including fall pulse
    plt.fill_between([0, famed_x-.5*fadur_x, famed_x, famed_x+.5*fadur_x, wsmed_x], [ds25_y, ds25_y, fa25_y, ds25_y, ds25_y], [ds75_y, ds75_y, fa75_y,ds75_y,ds75_y], facecolor='grey')
    # vertical fill around wet season start (to capture variation in start timing)
    plt.fill_betweenx([ds25_y, wsmed_y], [ws25_x], [ws75_x], facecolor='grey')
    # fill around wet season magnitude until start of spring
    plt.fill_between([ws25_x, pre_spmed_x], [ws25_y], [ws75_y], facecolor='grey')
    # fill around rising limb of spring, arbitrary starting point at 60 days before 
    plt.fill_between([pre_spmed_x, spmed_x], [ws25_y, sp25_y], [ws75_y, sp75_y], facecolor='grey')
    # fill around falling limb of spring, chose 30 days to set SP_ROC slope (with adjustments)
    plt.fill_between([spmed_x, sp_rocmed_x], [sp25_y, sp_roc25_y], [sp75_y, sp_roc75_y], facecolor='grey')
    # fill around falling limb from spring ROC to start of dry season
    plt.fill_between([sp_rocmed_x, dsmed_x], [sp_roc25_y, ds25_y], [sp_roc75_y, ds75_y], facecolor='grey')
    # fill from dry season to end of water year
    plt.fill_between([dsmed_x, 366], [ds25_y], [ds75_y], facecolor='grey')

    plt.show()
    import pdb; pdb.set_trace()
    # plt.plot((x1, x2), (y1, y2), 'k-')
    # import pdb; pdb.set_trace()
    plt.show()

