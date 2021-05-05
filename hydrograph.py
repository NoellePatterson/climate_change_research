import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def site_hydrograph(ffc_data, rh_data):
    import pdb; pdb.set_trace()
    for gage_index, gage in enumerate(ffc_data):
        a=1


# narrow down for sites of interest
# average values across all models
# separate values into historic and future
# plot rh lines for med, 25/75th. 
# overlay critical values: timings and mags. 

def define_fill_points(year_type, percent, spmed_y, sp_rocmed_y):
    ws_x = year_type.loc['Wet_Tim_'+percent]
    pre_sp_x = year_type.loc['SP_Tim_'+percent]-60
    sp_x = year_type.loc['SP_Tim_'+percent]
    sp_roc_x = year_type.loc['SP_Tim_'+percent]+30
    ds_x = year_type.loc['DS_Tim_'+percent]

    # all 25th percentile fill y vals
    fa_y = year_type.loc['FA_Mag_'+percent]
    ds_y = year_type.loc['DS_Mag_50_'+percent]
    ws_y = year_type.loc['Wet_BFL_Mag_50_'+percent]
    sp_y = year_type.loc['SP_Mag_'+percent]
    sp_roc_y = sp_rocmed_y - (spmed_y - sp_y)
    # if end of spring recession drops too low based on original magnitude settings, reassign magnitude to dry season value
    if int(percent) < 50:
        if sp_roc_y < ds_y:
            sp_roc_y = ds_y
    # if fill value is greater than median
    elif int(percent) > 50:
        sp_roc_y = sp_rocmed_y + (sp_y - spmed_y)

    return ws_x, pre_sp_x, sp_x, sp_roc_x, ds_x, fa_y, ds_y, ws_y, sp_y, sp_roc_y

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
    percentiles = [['_10', .10], ['_25', .25], ['_med', .5], ['_75', .75], ['_90', .9]]
    for label, quantile in percentiles:
        for metric in metrics:
            ffc_vals.loc[metric+label, 'all_years'] = control_transpose[metric].quantile(quantile) 
            ffc_vals.loc[metric+label, 'dry'] = dry_q[metric].quantile(quantile) 
            ffc_vals.loc[metric+label, 'mod'] = mod_q[metric].quantile(quantile) 
            ffc_vals.loc[metric+label, 'wet'] = wet_q[metric].quantile(quantile)
    
    # 3. plot out hydrograph with 25/75 lines

    ################################## Option 1: DRH style background ####################################################
    # # Find the control plotting data from the list of simulations
    # for index, sim in enumerate(rh_data):
    #     if sim['name'] == 'DT0DP1':
    #         control_index = index
    
    # plot_data = rh_data[control_index]
    # percentiles = [10, 25, 50, 75, 90]
    # percentile_keys = ['ten', 'twenty_five', 'fifty', 'seventy_five', 'ninety']

    # rh = {}
    # for index, percentile in enumerate(percentile_keys):
    #     rh[percentile] = []
    # for row_index, _ in enumerate(plot_data['data'].iloc[:,0]): # loop through each row, 366 total
    #     # loop through all five percentiles
    #     for index, percentile in enumerate(percentiles): 
    #         # calc flow percentiles across all years for each row of flow matrix
    #         flow_row = pd.to_numeric(plot_data['data'].iloc[row_index, :], errors='coerce')
    #         rh[percentile_keys[index]].append(np.nanpercentile(flow_row, percentile))
        
    # plt.rc('ytick', labelsize=5) 
    # plt.subplot()
    # name = plot_data['name']
    # # make plot

    # x = np.arange(0,366,1)
    # ax = plt.plot(rh['ten'], color = 'darkgrey', label = "10%") # original color is navy
    # # plt.plot(rh['twenty_five'], color = 'blue', label = "25%")
    # plt.plot(rh['fifty'], color='red', label = "50%")
    # # plt.plot(rh['seventy_five'], color = 'blue', label = "75%")
    # plt.plot(rh['ninety'], color = 'darkgrey', label = "90%") # original color is navy
    # # plt.fill_between(x, rh['ten'], rh['twenty_five'], color='powderblue')
    # # plt.fill_between(x, rh['twenty_five'], rh['fifty'], color='powderblue')
    # # plt.fill_between(x, rh['fifty'], rh['seventy_five'], color='powderblue')
    # # plt.fill_between(x, rh['seventy_five'], rh['ninety'], color='powderblue')
    # plt.fill_between(x, rh['ten'], rh['fifty'], color='lightgrey')
    # plt.fill_between(x, rh['fifty'], rh['ninety'], color='lightgrey')

    # plt.title('Control - {} - Dry flow years'.format(name), size=9)
    # plt.ylabel('Flow (cfs)')
    # tick_spacing = [0, 30.5, 61, 91.5, 122, 152.5, 183, 213.5, 244, 274.5, 305, 335.5]
    # tick_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    # plt.xticks(tick_spacing, tick_labels)
    # plt.tick_params(axis='y', which='major', pad=1)
    # plt.axhline(y=0, ls=':', color='darkgrey')
    ###################################################################################################################

    ################################## Option 2: Yearly trace 'spaghetti plots ########################################
    for index, sim in enumerate(rh_data):
        if sim['name'] == 'DT0DP1':
            control_index = index
            name = sim['name']
    plot_data = rh_data[control_index]['data']
    plt.rc('ytick') 
    plt.subplot()
    x = np.arange(0,366,1)
    greyscale = np.linspace(.35, .92, 100)
    # import pdb; pdb.set_trace()
    # make plot
    for index, flow in enumerate(plot_data):
        y_plot = pd.to_numeric(plot_data.iloc[:,index], errors='coerce')
        plt.plot(y_plot, linewidth=.6, c=str(greyscale[index]), zorder=1)
        # try one bold hydrograph for reference
        if index == 6: # use 2 for dry years, 63 for moderate/all years, 6 for wet years
            plt.plot(y_plot, linewidth=.6, color='black', zorder=2)
    plt.title('Control - {} - Wet years'.format(name), size=9)
    plt.ylabel('Flow (cfs)') 
    tick_spacing = [0, 30.5, 61, 91.5, 122, 152.5, 183, 213.5, 244, 274.5, 305, 335.5]
    tick_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    plt.xticks(tick_spacing, tick_labels)

    ###################################################################################################################
    
    # 4. Plot out boxes based on FFC metric vals. 
    years = 'wet' # options are: 'all_years', 'dry', 'mod', 'wet' 
    year_type = ffc_vals[years]

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
    x_med = [0, famed_x-.5*fadur_x,famed_x,famed_x+.5*fadur_x, wsmed_x,wsmed_x, pre_spmed_x, spmed_x, sp_rocmed_x, dsmed_x, 366]

    famed_y = year_type.loc['FA_Mag_med']
    dsmed_y = year_type.loc['DS_Mag_50_med']
    wsmed_y = year_type.loc['Wet_BFL_Mag_50_med']
    # change peak value to lowest percentile for dry years
    if years == 'dry':
        peakmed_y = year_type.loc['Peak_2_med']
    else:
        peakmed_y = year_type.loc['Peak_5_med']
    spmed_y = year_type.loc['SP_Mag_med']
    sp_rocmed_y = year_type.loc['SP_Mag_med']-30*year_type.loc['SP_ROC_med']*year_type.loc['SP_Mag_med']
    y_med = [dsmed_y, dsmed_y,famed_y,dsmed_y, dsmed_y,wsmed_y, wsmed_y, spmed_y, sp_rocmed_y, dsmed_y, dsmed_y]
    plt.plot(x_med, y_med, 'k--', zorder=4)

    '''
    Plot error bound fill to show 10-90 range of FFC values
    '''

    ws_x_10, pre_sp_x_10, sp_x_10, sp_roc_x_10, ds_x_10, fa_y_10, ds_y_10, ws_y_10, sp_y_10, sp_roc_y_10 = define_fill_points(year_type, '10', spmed_y, sp_rocmed_y)
    ws_x_25, pre_sp_x_25, sp_x_25, sp_roc_x_25, ds_x_25, fa_y_25, ds_y_25, ws_y_25, sp_y_25, sp_roc_y_25 = define_fill_points(year_type, '25', spmed_y, sp_rocmed_y)
    ws_x_75, pre_sp_x_75, sp_x_75, sp_roc_x_75, ds_x_75, fa_y_75, ds_y_75, ws_y_75, sp_y_75, sp_roc_y_75 = define_fill_points(year_type, '75', spmed_y, sp_rocmed_y)
    ws_x_90, pre_sp_x_90, sp_x_90, sp_roc_x_90, ds_x_90, fa_y_90, ds_y_90, ws_y_90, sp_y_90, sp_roc_y_90 = define_fill_points(year_type, '90', spmed_y, sp_rocmed_y)

    x_10 = [0, ws_x_10, ws_x_10, pre_sp_x_10, sp_x_10, sp_roc_x_10, ds_x_10]
    x_25 = [0, ws_x_25, ws_x_25, pre_sp_x_25, sp_x_25, sp_roc_x_25, ds_x_25]
    x_75 = [0, ws_x_75, ws_x_75, pre_sp_x_75, sp_x_75, sp_roc_x_75, ds_x_75]
    x_90 = [0, ws_x_90, ws_x_90, pre_sp_x_90, sp_x_90, sp_roc_x_90, ds_x_90]

    y_10 = [ds_y_10, ds_y_10, ws_y_10, ws_y_10, sp_y_10, sp_roc_y_10, ds_y_10]
    y_25 = [ds_y_25, ds_y_25, ws_y_25, ws_y_25, sp_y_25, sp_roc_y_25, ds_y_25]
    y_75 = [ds_y_75, ds_y_75, ws_y_75, ws_y_75, sp_y_75, sp_roc_y_75, ds_y_75]
    y_90 = [ds_y_90, ds_y_90, ws_y_90, ws_y_90, sp_y_90, sp_roc_y_90, ds_y_90]

    # plot horizontal line to show peak val magnitude
    plt.plot([ws_x_25, pre_spmed_x], [peakmed_y, peakmed_y], 'k--', zorder=4)

    '''
    Use fill between function to plot 25/75 percentile bounds
    '''
    # fill from start of year to start of wet season, including fall pulse
    plt.fill_between([0, famed_x-.5*fadur_x, famed_x, famed_x+.5*fadur_x, wsmed_x], [ds_y_25, ds_y_25, fa_y_25, ds_y_25, ds_y_25], [ds_y_75, ds_y_75, fa_y_75,ds_y_75,ds_y_75], facecolor='blue', zorder=3, alpha=0.65)
    # vertical fill around wet season start (to capture variation in start timing)
    plt.fill_betweenx([ds_y_25, ws_y_75], [ws_x_25], [ws_x_75], facecolor='blue', zorder=3, alpha=0.65)
    # fill around wet season magnitude until start of spring
    plt.fill_between([ws_x_75, pre_spmed_x], [ws_y_25], [ws_y_75], facecolor='blue', zorder=3, alpha=0.65)
    # fill around rising limb of spring, arbitrary starting point at 60 days before 
    plt.fill_between([pre_spmed_x, spmed_x], [ws_y_25, sp_y_25], [ws_y_75, sp_y_75], facecolor='blue', zorder=3, alpha=0.65)
    # fill around falling limb of spring, chose 30 days to set SP_ROC slope (with adjustments)
    plt.fill_between([spmed_x, sp_rocmed_x], [sp_y_25, sp_roc_y_25], [sp_y_75, sp_roc_y_75], facecolor='blue', zorder=3, alpha=0.65)
    # fill around falling limb from spring ROC to start of dry season
    plt.fill_between([sp_rocmed_x, dsmed_x], [sp_roc_y_25, ds_y_25], [sp_roc_y_75, ds_y_75], facecolor='blue', zorder=3, alpha=0.65)
    # fill from dry season to end of water year
    plt.fill_between([dsmed_x, 366], [ds_y_25], [ds_y_75], facecolor='blue', zorder=3, alpha=0.65)

    '''
    Use fill between function to plot 10/90 percentile bounds
    '''
    # fill from start of year to start of wet season, including fall pulse
    plt.fill_between([0, famed_x-.5*fadur_x, famed_x, famed_x+.5*fadur_x, wsmed_x], [ds_y_10, ds_y_10, fa_y_10, ds_y_10, ds_y_10], [ds_y_90, ds_y_90, fa_y_90,ds_y_90,ds_y_90], facecolor='blue', zorder=2, alpha=0.45)
    # vertical fill around wet season start (to capture variation in start timing)
    plt.fill_betweenx([ds_y_10, ws_y_90], [ws_x_10], [ws_x_90], facecolor='blue', zorder=2, alpha=0.45)
    # fill around wet season magnitude until start of spring
    plt.fill_between([ws_x_90, pre_spmed_x], [ws_y_10], [ws_y_90], facecolor='blue', zorder=2, alpha=0.45)
    # fill around rising limb of spring, arbitrary starting point at 60 days before 
    plt.fill_between([pre_spmed_x, spmed_x], [ws_y_10, sp_y_10], [ws_y_90, sp_y_90], facecolor='blue', zorder=2, alpha=0.45)
    # fill around falling limb of spring, chose 30 days to set SP_ROC slope (with adjustments)
    plt.fill_between([spmed_x, sp_rocmed_x], [sp_y_10, sp_roc_y_10], [sp_y_90, sp_roc_y_90], facecolor='blue', zorder=2, alpha=0.45)
    # fill around falling limb from spring ROC to start of dry season
    plt.fill_between([sp_rocmed_x, dsmed_x], [sp_roc_y_10, ds_y_10], [sp_roc_y_90, ds_y_90], facecolor='blue', zorder=2, alpha=0.45)
    # fill from dry season to end of water year
    plt.fill_between([dsmed_x, 366], [ds_y_10], [ds_y_90], facecolor='blue', zorder=2, alpha=0.45)

    plt.show()
    # import pdb; pdb.set_trace()

    '''
    Output table of hydrograph values
    '''
    year_types = ['all_years', 'dry', 'mod', 'wet']
    output_table = pd.DataFrame(columns = ['val'])
    metrics = [ 'DS_Mag_50', 'FA_Mag', 'Wet_Tim', 'Wet_BFL_Mag_50', 'Peak_5','Peak_Dur_5', 'SP_Tim', 'SP_Mag', 'SP_ROC', 'DS_Tim']
    percentiles = [['_10', .10], ['_25', .25], ['_med', .5], ['_75', .75], ['_90', .9]]
    for year_type in year_types:
        for metric in metrics:
            for label, quantile in percentiles:
                if year_type == 'all_years':
                    output_table.loc[metric+label, 'val'] = control_transpose[metric].quantile(quantile) 
                elif year_type == 'dry':
                    output_table.loc[metric+label, 'dry'] = dry_q[metric].quantile(quantile) 
                elif year_type == 'mod':
                    output_table.loc[metric+label, 'mod'] = mod_q[metric].quantile(quantile) 
                elif year_type == 'wet':
                    output_table.loc[metric+label, 'wet'] = wet_q[metric].quantile(quantile)
        output_table.to_csv('hydrograph_metrics_'+year_type+'.csv')
    import pdb; pdb.set_trace()








