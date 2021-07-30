import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12.5})
from functools import reduce


def merced_models_hydrograph(ffc_data, rh_data):
    # Dry yr=2008, avg yr=1955, wet yr=1998
    ctrl = {}
    oat_t = {}
    oat_pwet = {}
    oat_pdry = {}
    oat_s = {}
    oat_e = {}
    oat_i = {}
    for model_index, model in enumerate(rh_data):
        if model['name'] == 'SACSMA_CTR_T0P0S0E0I0':
            data = model['data'].apply(pd.to_numeric, errors='coerce')
            ctrl['rh'] = data
        elif model['name'] == 'SACSMA_OATT_T5P0S0E0I0':
            data = model['data'].apply(pd.to_numeric, errors='coerce')
            oat_t['rh'] = data
        elif model['name'] == 'SACSMA_OATP_T0P30S0E0I0':
            data = model['data'].apply(pd.to_numeric, errors='coerce')
            oat_pwet['rh'] = data
        elif model['name'] == 'SACSMA_OATP_T0P-30S0E0I0':
            data = model['data'].apply(pd.to_numeric, errors='coerce')
            oat_pdry['rh'] = data
        elif model['name'] == 'SACSMA_OATS_T0P0S5E0I0':
            data = model['data'].apply(pd.to_numeric, errors='coerce')
            oat_s['rh'] = data
        elif model['name'] == 'SACSMA_OATE_T0P0S0E5I0':
            data = model['data'].apply(pd.to_numeric, errors='coerce')
            oat_e['rh'] = data
        elif model['name'] == 'SACSMA_OATI_T0P0S0E0I5':
            data = model['data'].apply(pd.to_numeric, errors='coerce')
            oat_i['rh'] = data
    for model_index, model in enumerate(ffc_data):
        if model['gage_id'] == 'SACSMA_CTR_T0P0S0E0I0':
            data = model['ffc_metrics'].apply(pd.to_numeric, errors='coerce')
            ctrl['ffc'] = data
        elif model['gage_id'] == 'SACSMA_OATT_T5P0S0E0I0':
            data = model['ffc_metrics'].apply(pd.to_numeric, errors='coerce')
            oat_t['ffc'] = data
        elif model['gage_id'] == 'SACSMA_OATP_T0P30S0E0I0':
            data = model['ffc_metrics'].apply(pd.to_numeric, errors='coerce')
            oat_pwet['ffc'] = data
        elif model['gage_id'] == 'SACSMA_OATP_T0P-30S0E0I0':
            data = model['ffc_metrics'].apply(pd.to_numeric, errors='coerce')
            oat_pdry['ffc'] = data
        elif model['gage_id'] == 'SACSMA_OATS_T0P0S5E0I0':
            data = model['ffc_metrics'].apply(pd.to_numeric, errors='coerce')
            oat_s['ffc'] = data
        elif model['gage_id'] == 'SACSMA_OATE_T0P0S0E5I0':
            data = model['ffc_metrics'].apply(pd.to_numeric, errors='coerce')
            oat_e['ffc'] = data
        elif model['gage_id'] == 'SACSMA_OATI_T0P0S0E0I5':
            data = model['ffc_metrics'].apply(pd.to_numeric, errors='coerce')
            oat_i['ffc'] = data
    # import pdb; pdb.set_trace()
    # Create plot canvas for dry, avg, wet
    dry_fig, ax = plt.subplots()
    def get_plotlines(year, ctrl, oat_t, oat_pwet, oat_pdry, oat_s, oat_e, oat_i):
        ctrl_plot = ctrl['rh'][year]
        oat_t_plot = oat_t['rh'][year]
        oat_pwet_plot = oat_pwet['rh'][year]
        oat_pdry_plot = oat_pdry['rh'][year]
        oat_s_plot = oat_s['rh'][year]
        oat_e_plot = oat_e['rh'][year]
        oat_i_plot = oat_i['rh'][year]
        plot_lines = [ctrl_plot, oat_t_plot, oat_pwet_plot, oat_pdry_plot, oat_s_plot, oat_e_plot, oat_i_plot]
        return(plot_lines)
    # dry year = 2008, avg = 1955, wet = 1998
    plot_lines_dry = get_plotlines('1962', ctrl, oat_t, oat_pwet, oat_pdry, oat_s, oat_e, oat_i)
    colors = ['black', 'red', 'blue', 'lightblue', 'green', 'darkorange', 'gold']
    labels = ['Control', 'Temperature', 'Precipitation - wet', 'Precipitation - dry', 'Seasonal intensity', 'Event intensity', 'Interannual intensity']
    for index, plot_line in enumerate(plot_lines_dry):
        ax.plot(plot_line, color=colors[index], label=labels[index], alpha=0.6)
    plt.legend(fontsize=10)
    plt.show()
    # import pdb; pdb.set_trace()

    # plot models (specific years) onto each canvas: cntl and OAT extremes
    
    return 


def site_hydrograph(ffc_data, rh_data):
    # narrow down for sites of interest
    def get_site_data(dataset, search_key, data_key, rcp):
        macclure = []
        battle = []
        englebright = []
        for site_index, site in enumerate(dataset):
            # error in results having to do with data as string type in upload!!!!
            if rcp in site['model_name']:
                if site[search_key] == 'I20____Lake_McClure_Inflow_calsim_and_wytypes':
                    data = site[data_key].apply(pd.to_numeric, errors='coerce')
                    macclure.append(data)
                elif site[search_key] == 'I10803_Battle_Creek_Inflow_to_Sacramento_River_calsim':
                    data = site[data_key].apply(pd.to_numeric, errors='coerce')
                    battle.append(data)
                elif site[search_key] == '11418000_Englebright_Stern_and_wytypes':
                    data = site[data_key].apply(pd.to_numeric, errors='coerce')
                    englebright.append(data)
        return(macclure, battle, englebright)

    # replace all Nones with row avg, so average across all df's will work
    def site_hydrograph_plotter(site_ffc, site_rh):
        # take avg of models for hist/fut metrics and for rh
        metrics_list = site_ffc[0].index
        all_models_hist = []
        all_models_fut = []
        rh_all_models_hist = []
        rh_all_models_fut = []
        for model in site_ffc:
            metrics_hist = model.iloc[:,0:65].mean(axis=1) # years 1950-2015
            metrics_fut = model.iloc[:,85:150].mean(axis=1) # years 2035-2100
            all_models_hist.append(metrics_hist)
            all_models_fut.append(metrics_fut)
        for model in site_rh:
            rh_all_models_hist.append(model.iloc[:, 0:65]) # 1950-2015
            rh_all_models_fut.append(model.iloc[:, 85:150]) # 2035-2100

        array_hist = np.array(all_models_hist)
        avg_hist = np.nanmean(array_hist, axis=0)
        array_fut = np.array(all_models_fut)
        avg_fut = np.nanmean(array_fut, axis=0)
        final_hist_metrics = pd.DataFrame(data=avg_hist, index = metrics_list)
        final_fut_metrics = pd.DataFrame(data=avg_fut, index = metrics_list)

        for model_index, model in enumerate(site_rh):  
            site_rh[model_index] = site_rh[model_index].replace('None', np.nan)
        site_rh_avg = pd.DataFrame(0, index=site_rh[0].index, columns = site_rh[0].columns)
        for model in site_rh:
            site_rh_avg = site_rh_avg.add(model.apply(pd.to_numeric))
        site_rh_avg = site_rh_avg.divide(10)   

        site_rh_hist = site_rh_avg.iloc[:, 0:65] # 1950-2015
        site_rh_fut = site_rh_avg.iloc[:, 85:150] # 2035-2100

        rh_hist = {}
        rh_fut = {}
        percentile_keys = ['twenty_five', 'fifty', 'seventy_five']
        percentiles = [25, 50, 75]
        for index, percentile in enumerate(percentile_keys):
            rh_hist[percentile] = []
            rh_fut[percentile] = []
        for row_index, _ in enumerate(site_rh_hist.iloc[:,0]): # loop through each row, 366 total
                # loop through all 3 percentiles
                for index, percentile in enumerate(percentiles): 
                    # calc flow percentiles across all years for each row of flow matrix
                    flow_row_hist = pd.to_numeric(site_rh_hist.iloc[row_index, :], errors='coerce')
                    rh_hist[percentile_keys[index]].append(np.nanpercentile(flow_row_hist, percentile))
                    flow_row_fut = pd.to_numeric(site_rh_fut.iloc[row_index, :], errors='coerce')
                    rh_fut[percentile_keys[index]].append(np.nanpercentile(flow_row_fut, percentile))
        np.nanmax(pd.to_numeric(site_rh[7].iloc[:,100]))
        fig, ax = plt.subplots()
        x = np.arange(0,366,1)
        month_ticks = [0,32,60,91,121,152,182,213,244,274,305,335]
        month_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
        ax.plot(rh_hist['fifty'], color = 'navy', label = "Historic (1950-2015)", linewidth=2)
        plt.fill_between(x, rh_hist['twenty_five'], rh_hist['fifty'], color='powderblue', alpha=.5)
        plt.fill_between(x, rh_hist['fifty'], rh_hist['seventy_five'], color='powderblue', alpha=.5)

        ax.plot(rh_fut['fifty'], color = 'darkred', label = "Future (2035-2100)", linewidth=2)
        plt.fill_between(x, rh_fut['twenty_five'], rh_fut['fifty'], color='lightpink', alpha=.5)
        plt.fill_between(x, rh_fut['fifty'], rh_fut['seventy_five'], color='lightpink', alpha=.5)
        
        # add plot anotations using ffc metrics
        ds_tim_fut = np.nanmean(final_fut_metrics.loc['DS_Tim'])
        sp_tim_fut = np.nanmean(final_fut_metrics.loc['SP_Tim'])
        wet_tim_fut = np.nanmean(final_fut_metrics.loc['Wet_Tim'])
        fa_tim_fut = np.nanmean(final_fut_metrics.loc['FA_Tim'])
        ds_mag_fut = np.nanmean(final_fut_metrics.loc['DS_Mag_50'])
        wet_mag_fut = np.nanmean(final_fut_metrics.loc['Wet_BFL_Mag_50'])

        ds_tim_hist = np.nanmean(final_hist_metrics.loc['DS_Tim'])
        sp_tim_hist = np.nanmean(final_hist_metrics.loc['SP_Tim'])
        wet_tim_hist = np.nanmean(final_hist_metrics.loc['Wet_Tim'])
        fa_tim_hist = np.nanmean(final_hist_metrics.loc['FA_Tim'])
        ds_mag_hist = np.nanmean(final_hist_metrics.loc['DS_Mag_50'])
        wet_mag_hist = np.nanmean(final_hist_metrics.loc['Wet_BFL_Mag_50'])
        # np.nanmean(site_ffc_fut.loc['SP_Mag'])
        # np.nanmean(site_ffc_hist.loc['SP_Mag'])

        plt.vlines([ds_tim_hist, sp_tim_hist, wet_tim_hist, fa_tim_hist], ymin=0, ymax= max(rh_fut['seventy_five']), color='navy', alpha=.75)
        plt.vlines([ds_tim_fut, sp_tim_fut, wet_tim_fut, fa_tim_fut], ymin=0, ymax= max(rh_fut['seventy_five']), color='darkred', alpha=.75)
        plt.hlines([ds_mag_hist], xmin=ds_tim_fut, xmax=366, color='navy', alpha=.65)
        plt.hlines([wet_mag_hist], xmin=wet_tim_hist, xmax=sp_tim_hist, color='navy', alpha=.65)
        plt.hlines([ds_mag_fut], xmin=ds_tim_fut, xmax=366, color='darkred', alpha=.65)
        plt.hlines([wet_mag_fut], xmin=wet_tim_fut, xmax=sp_tim_fut, color='darkred', alpha=.65)

        ax.legend(loc='upper left')
        # ax.grid(which="major", axis='y')
        ax.set_ylabel('Flow (cfs)')
        plt.xticks(month_ticks, month_labels)
        # plt.title('Merced River at Lake McClure (Central)')
        plt.title('Yuba River below Englebright Dam (Southern)')
        # plt.title('Battle Creek (Northern)')
        plt.show()
        import pdb; pdb.set_trace() 

    macclure_ffc, battle_ffc, englebright_ffc = get_site_data(ffc_data, 'gage_id', 'ffc_metrics', '85')
    macclure_rh, battle_rh, englebright_rh = get_site_data(rh_data, 'name', 'data', '85')
    site_hydrograph_plotter(englebright_ffc, englebright_rh)



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








