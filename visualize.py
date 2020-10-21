import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

'''
Print DRH plots for time series flow data
'''

def plot_drh(drh_data):
    # reorder list so that simulations plot out in a logical order
    sim_order = [25, 18, 28, 29, 26, 19, 20, 23, 13, 27, 21, 24, 10, 12, 22, 0, 9, 11, 17, 7, 3, 2, 14, 6, 16, 8, 4, 1, 15, 5]
    drh_data = [drh_data[i] for i in sim_order]
    for index, simulation in enumerate(drh_data):
        plt.rc('ytick', labelsize=5) 
        plt.subplot(5, 6, index+1)
        name = simulation['name']
        data = simulation['data']
        data = data.transpose()
        # make plot
        plt.title('{}'.format(name), size=9)
        
        plt.xticks([])
        plt.tick_params(axis='y', which='major', pad=1)

        x = np.arange(0,366,1)
        ax = plt.plot(data['ten'], color = 'navy', label = "10%")
        plt.plot(data['twenty_five'], color = 'blue', label = "25%")
        plt.plot(data['fifty'], color='red', label = "50%")
        plt.plot(data['seventy_five'], color = 'blue', label = "75%")
        plt.plot(data['ninty'], color = 'navy', label = "90%")
        plt.plot(data['min'], color = 'black', label = 'min', lw=1)
        plt.plot(data['max'], color = 'black', label = 'max', lw=1)
        plt.fill_between(x, data['ten'], data['twenty_five'], color='powderblue')
        plt.fill_between(x, data['twenty_five'], data['fifty'], color='powderblue')
        plt.fill_between(x, data['fifty'], data['seventy_five'], color='powderblue')
        plt.fill_between(x, data['seventy_five'], data['ninty'], color='powderblue')

        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=7, borderaxespad = .9, fontsize='small', labelspacing=.2, columnspacing=1, markerscale=.5)
        # plt.ylabel("Daily Flow/Average Annual Flow")

    plt.show()
  
    return None

def plot_rh(rh_data):
    # reorder list so that simulations plot out in a logical order
    sim_order = [21, 20, 25, 17, 19, 23, 15, 26, 1, 22, 28, 13, 10, 2, 18, 5, 0, 12, 4, 6, 8, 11, 7, 3, 14, 29, 9, 24, 27, 16]
    rh_data = [rh_data[i] for i in sim_order]

    percentiles = [10, 25, 50, 75, 90]
    percentile_keys = ['ten', 'twenty_five', 'fifty', 'seventy_five', 'ninety']

    for sim_index, simulation in enumerate(rh_data):
        rh = {}
        for index, percentile in enumerate(percentile_keys):
            rh[percentile] = []
        # within each simulation, pull out flow percentiles from each row
        for row_index, _ in enumerate(simulation['data'].iloc[:,0]): # loop through each row, 366 total
            # loop through all five percentiles
            for index, percentile in enumerate(percentiles): 
                # calc flow percentiles across all years for each row of flow matrix
                flow_row = pd.to_numeric(simulation['data'].iloc[row_index, :], errors='coerce')
                rh[percentile_keys[index]].append(np.nanpercentile(flow_row, percentile))
        
        plt.rc('ytick', labelsize=5) 
        plt.subplot(5, 6, sim_index+1)
        name = simulation['name']
        # make plot
        plt.title('{}'.format(name), size=9)
        
        plt.xticks([])
        plt.tick_params(axis='y', which='major', pad=1)

        x = np.arange(0,366,1)
        ax = plt.plot(rh['ten'], color = 'navy', label = "10%")
        plt.plot(rh['twenty_five'], color = 'blue', label = "25%")
        plt.plot(rh['fifty'], color='red', label = "50%")
        plt.plot(rh['seventy_five'], color = 'blue', label = "75%")
        plt.plot(rh['ninety'], color = 'navy', label = "90%")
        plt.fill_between(x, rh['ten'], rh['twenty_five'], color='powderblue')
        plt.fill_between(x, rh['twenty_five'], rh['fifty'], color='powderblue')
        plt.fill_between(x, rh['fifty'], rh['seventy_five'], color='powderblue')
        plt.fill_between(x, rh['seventy_five'], rh['ninety'], color='powderblue')
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=7, borderaxespad = .9, fontsize='small', labelspacing=.2, columnspacing=1, markerscale=.5)
        
    plt.show()

def line_plots(ffc_data):
    # gather plots of only one precip range (or temp range)
    p_08 = []
    p_09 = []
    p_1 = []
    p_11 = []
    p_12 = []
    p_13 = []
    for index, simulation in enumerate(ffc_data):
        # categorize all simulations into groups based on precip level. Should be 6 simulations per group. 
        if simulation['gage_id'][4:] == 'P0.8':
            p_08.append(simulation)
        if simulation['gage_id'][4:] == 'P0.9':
            p_09.append(simulation)
        if simulation['gage_id'][4:] == 'P1':
            p_1.append(simulation)
        if simulation['gage_id'][4:] == 'P1.1':
            p_11.append(simulation)
        if simulation['gage_id'][4:] == 'P1.2':
            p_12.append(simulation)
        if simulation['gage_id'][4:] == 'P1.3':
            p_13.append(simulation)
    p_levels = [p_08, p_09, p_1, p_11, p_12, p_13]
    for p_level in p_levels:
        p_name = p_level[0]['gage_id'][4:]
        x = pd.to_numeric(simulation['ffc_metrics'].columns)
        fig = plt.figure(figsize=(10, 4))
        plt.subplot(1,1,1)
        for simulation in p_level:
            name = simulation['gage_id']
            if name[:3] == 'DT0':
                color = '#FFEC19'
            elif name[:3] == 'DT1':
                color = '#FFC100'
            elif name[:3] == 'DT2':
                color = '#FF9800'
            elif name[:3] == 'DT3':
                color = '#FF5607'
            elif name[:3] == 'DT4':
                color = '#F6412D'
            # import pdb; pdb.set_trace()
            y = pd.to_numeric(simulation['ffc_metrics'].loc['SP_Tim'], errors='coerce')
            plt.plot(x, y, label=name, color=color)
            # plt.show()
        # pull out control sim directly; it is the 27th in the group. 
        control = pd.to_numeric(ffc_data[27]['ffc_metrics'].loc['SP_Tim'], errors='coerce') 
        plt.plot(x, control, '--', label='DT0P1_control', color='black', linewidth=.8)
        plt.title('Spring Recession Timing')
        plt.legend(fancybox=True, borderaxespad = .9, fontsize='small', labelspacing=.2, columnspacing=1, markerscale=.5)
        fig.savefig('data_outputs/plots/lines/spring_tim_{}_alltemps.pdf'.format(p_name))

def scatterplot(ffc_data):
    # narrow down to all six DT4 results plus DT0P1 control
    dt0 = []
    dt1 = []
    dt2 = []
    dt3 = []
    dt4 = []
    for index, simulation in enumerate(ffc_data):
        if simulation['gage_id'][0:3] == 'DT0':
            dt0.append(simulation)
        if simulation['gage_id'][0:3] == 'DT1':
            dt1.append(simulation)
        if simulation['gage_id'][0:3] == 'DT2':
            dt2.append(simulation)
        if simulation['gage_id'][0:3] == 'DT3':
            dt3.append(simulation)
        if simulation['gage_id'][0:3] == 'DT4':
            dt4.append(simulation)
        if simulation['gage_id'] == 'DT0DP1':
            control_sim = simulation
    # import pdb; pdb.set_trace()
    mag_metric = 'Wet_BFL_Mag_50' # FA_Mag' , 'Wet_BFL_Mag_50' , 'SP_Mag' , DS_Mag_50
    time_metric = 'Wet_Tim' # 'FA_Tim' , 'Wet_Tim' , 'SP_Tim' , DS_Tim
    dsmag_control = np.nanmean(pd.to_numeric(control_sim['ffc_metrics'].loc[mag_metric], errors='coerce'))
    dstim_control = np.nanmean(pd.to_numeric(control_sim['ffc_metrics'].loc[time_metric], errors='coerce'))

    temp_sims = [dt0, dt1, dt2, dt3, dt4]
    for temp_sim in temp_sims:
        fig, ax = plt.subplots(figsize=(8,8))
        contl_x = pd.to_numeric(control_sim['ffc_metrics'].loc[time_metric], errors='coerce')
        contl_y = pd.to_numeric(control_sim['ffc_metrics'].loc[mag_metric], errors='coerce')
        ax.scatter(contl_x, contl_y, color='black', label='Control DT0DP1')
        ax.set_ylim(-100, 7500) # fall: max 8600 , wet: max 7500 , sp: max 60000 , dry: 700
        ax.set_xlim(-1, 185) # fall: (-1, 60) , wet: (-1, 185) , sp: (50, 350) , dry: (230, 395)
        

        # loop through the precip sims within each temp sim
        for sim_index, sim in enumerate(temp_sim):
            name = sim['gage_id']
            if name[-3:] == '0.8': 
                color = '#eb9800' # orange
            elif name[-3:] == '0.9': 
                color = '#ffe3b1' # light orange
            elif name[-3:] == 'DP1': 
                color = '#f6fbff' # very light blue
            elif name[-3:] == '1.1': 
                color = '#cfeaff' # light blue
            elif name[-3:] == '1.2':
                color = '#1e9bff' # medium blue
            elif name[-3:] == '1.3':
                color = '#005da8' # dark blue
            # import pdb; pdb.set_trace()
            x = pd.to_numeric(sim['ffc_metrics'].loc[time_metric], errors='coerce')
            y = pd.to_numeric(sim['ffc_metrics'].loc[mag_metric], errors='coerce')
            ax.scatter(x, y, color=color, edgecolors='black', label=name)
        plt.axhline(y=dsmag_control, ls='--', color='black', label='Average control timing')
        plt.axvline(x=dstim_control, ls=':', color='black', label='Average control magnitude')
        # import pdb; pdb.set_trace()
        
        # Attempt to order legend did not work, consider trying again later
        # handles, labels = ax.get_legend_handles_labels()
        # by_label = OrderedDict(zip(labels, handles))
        # plt.legend(labels, handles)

        plt.legend(fancybox=True, borderaxespad = .9, fontsize='small', labelspacing=.2, columnspacing=1, markerscale=.5)
        
        plt.ylabel("Wet Season Magnitude")
        plt.xlabel("Wet Season Timing")
        
        fig.savefig('data_outputs/plots/scatter/wet_tim_mag_{}.pdf'.format(temp_sim[0]['gage_id'][0:3]))
        # plt.show()
        # import pdb; pdb.set_trace()

