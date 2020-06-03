import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

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

def line_plots(ffc_data):
    # gather plots of only one precip range (or temp range)
    p_08 = []
    p_09 = []
    p_1 = []
    p_11 = []
    p_12 = []
    p_13 = []
    for index, simulation in enumerate(ffc_data):
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
            # import pdb; pdb.set_trace()
            y = pd.to_numeric(simulation['ffc_metrics'].loc['DS_Tim'], errors='coerce')
            plt.plot(x, y, label=name)
            # plt.show()
        control = pd.to_numeric(ffc_data[27]['ffc_metrics'].loc['DS_Tim'], errors='coerce')
        plt.plot(x, y, '--', label='DT0P1_control', color='black', linewidth=.8)
        plt.title('Dry Season Timing')
        plt.legend(fancybox=True, borderaxespad = .9, fontsize='small', labelspacing=.2, columnspacing=1, markerscale=.5)
        fig.savefig('data_outputs/plots/dry_tim_{}_alltemps.pdf'.format(p_name))

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

    dsmag_control = np.nanmean(pd.to_numeric(control_sim['ffc_metrics'].loc['DS_Mag_50'], errors='coerce'))
    dstim_control = np.nanmean(pd.to_numeric(control_sim['ffc_metrics'].loc['DS_Tim'], errors='coerce'))

    scatter_colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink']
    fig, ax = plt.subplots(figsize=(8,8))
    contl_x = pd.to_numeric(control_sim['ffc_metrics'].loc['DS_Tim'], errors='coerce')
    contl_y = pd.to_numeric(control_sim['ffc_metrics'].loc['DS_Mag_50'], errors='coerce')
    ax.scatter(contl_x, contl_y, color='black', label='Control DT0DP1')

    for sim_index, sim in enumerate(dt3):
        name = sim['gage_id']
        # import pdb; pdb.set_trace()
        x = pd.to_numeric(sim['ffc_metrics'].loc['DS_Tim'], errors='coerce')
        y = pd.to_numeric(sim['ffc_metrics'].loc['DS_Mag_50'], errors='coerce')
        ax.scatter(x, y, color=scatter_colors[sim_index], label=name)
    plt.axhline(y=dsmag_control, ls='--', color='black', label='Average control timing')
    plt.axvline(x=dstim_control, ls=':', color='black', label='Average control magnitude')
    plt.legend(fancybox=True, borderaxespad = .9, fontsize='small', labelspacing=.2, columnspacing=1, markerscale=.5)
    plt.ylabel("Dry Season Magnitude")
    plt.xlabel("Dry Season Timing")
    fig.savefig('data_outputs/plots/scatter/dry_tim_mag_DT3.pdf')
    plt.show()
    import pdb; pdb.set_trace()

