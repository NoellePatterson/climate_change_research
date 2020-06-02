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
    for index, simulation in enumerate(ffc_data):
        if simulation['gage_id'][4:] == 'P0.8':
            p_08.append(simulation)
        if simulation['gage_id'][4:] == 'P0.9':
            p_09.append(simulation)

    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1,1,1)
    for simulation in p_08:
        name = simulation['gage_id']
        print(name)
        x = np.arange(0, len(simulation['ffc_metrics'].iloc[0]))
        y = pd.to_numeric(simulation['ffc_metrics'].loc['DS_Tim'])
        # import pdb; pdb.set_trace()
        plt.plot(x, y, label=name)
        fig.savefig('data_outputs/plots/test.pdf')
        # plt.show()
    plt.title('Dry Season Timing')
    plt.legend(fancybox=True, borderaxespad = .9, fontsize='small', labelspacing=.2, columnspacing=1, markerscale=.5)
    fig.savefig('data_outputs/plots/dryseason_p08_alltemps.pdf')
    import pdb; pdb.set_trace()
    # plot lines in chrono order