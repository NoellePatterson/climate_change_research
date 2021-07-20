import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits import mplot3d
import seaborn as sns
import numpy as np
from collections import OrderedDict
# for import of data from seaborn
import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# # plot model scenarios
# models = pd.read_csv('data_inputs/model_scenarios_plot.csv')
# T = models['T']
# P = models['P']
# I = models['I']

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(T, P, I)
# ax.set_xlabel('Temperature')
# ax.set_ylabel('Precipitation Volume')
# ax.set_zlabel('Precipitation Intensity')
# plt.show()


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
    # sim_order = [21, 20, 25, 17, 19, 23, 15, 26, 1, 22, 28, 13, 10, 2, 18, 5, 0, 12, 4, 6, 8, 11, 7, 3, 14, 29, 9, 24, 27, 16]
    runs = ['SACSMA_T0P0S0E0I0', 'SACSMA_T0P0S1E0I0', 'SACSMA_T0P0S2E0I0', 'SACSMA_T0P0S3E0I0', 'SACSMA_T0P0S4E0I0', 'SACSMA_T0P0S5E0I0']
    percentiles = [10, 25, 50, 75, 90]
    percentile_keys = ['ten', 'twenty_five', 'fifty', 'seventy_five', 'ninety']
    for run_index, m_run in enumerate(runs):
        # import pdb; pdb.set_trace()
        current_run = False
        for index, model in enumerate(rh_data):
            if model['name'] == m_run:
                current_run = model
                break
        if not current_run:
            raise ValueError('Model run specified does not exist.')

        rh = {}
        for index, percentile in enumerate(percentile_keys):
            rh[percentile] = []
        # within each run, pull out flow percentiles from each row
        for row_index, _ in enumerate(current_run['data'].iloc[:,0]): # loop through each row, 366 total
            # loop through all five percentiles
            for index, percentile in enumerate(percentiles): 
                # calc flow percentiles across all years for each row of flow matrix
                flow_row = pd.to_numeric(current_run['data'].iloc[row_index, :], errors='coerce')
                rh[percentile_keys[index]].append(np.nanpercentile(flow_row, percentile))
        
        plt.rc('ytick', labelsize=5) 
        plt.subplot(6, 1, run_index+1)
        name = current_run['name']
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
    plt.suptitle('80th Percentile Intensity shifts')    
    plt.show()

def boxplot(ffc_data):
    # metric to plot
    metric = 'SP_Tim'
    # Select which runs to produce plots for
    runs = ['run1', 'run14', 'run15', 'run16', 'run17']
    boxplot_array = []
    # Arrange data from each run (for a specific metric or all metrics) into array
    for run_index, m_run in enumerate(runs):
        for index, model in enumerate(ffc_data):
            if model['gage_id'] == m_run:
                data = pd.to_numeric(model['ffc_metrics'].loc[metric], errors='coerce')
                # remove nans so they don't mess up boxplot
                data = data[~np.isnan(data)] 
                boxplot_array.append(data)
                # boxplot_array.append(pd.to_numeric(model['ffc_metrics'].loc[metric], errors='coerce'))
                break
    # Create boxplot with array of data from runs.  
    plt.boxplot(boxplot_array, labels=runs)
    plt.title(metric+" - 80th Percentile Shifts")
    plt.savefig('data_outputs/plots/boxplots/{}_runs14-17.pdf'.format(metric))
    # plt.show()
    # import pdb; pdb.set_trace() 

def jitterplot(ffc_data):
    # extract model data to plot
    control_id = 'run1'
    exp_id = 'run265' # 'run265' 'run1115' 
    exp_id2 = 'run1115' 
    exp_id3 = 'run26' # intensity only
    metrics = ['DS_Tim']# ['Avg', 'Std'] ['DS_Mag_50', 'DS_Mag_90', 'Wet_BFL_Mag_10', 'Wet_BFL_Mag_50']# ['FA_Tim', 'Wet_Tim', 'SP_Tim', 'DS_Tim']
    y_label = 'Day of Water Year' # 'Day of Water Year' 'Flow, cfs'
    legend_labels = ['Control', 'Temp '+r'$\Delta$5 and intensity', 'Temp '+r'$\Delta$5 only', 'Intensity only']
    filename = 'all_models_dstim'
    # ymax = 13200 # timings:450 mags:13200 avg:8000 
    for model in ffc_data:
        if model['gage_id'] == control_id:
            control = model
        elif model['gage_id'] == exp_id:
            exp = model
        elif model['gage_id'] == exp_id2:
            exp2 = model
        elif model['gage_id'] == exp_id3:
            exp3 = model

    # dicts to only models/metrics being used.
    control = control['ffc_metrics'].loc[metrics, :]
    exp = exp['ffc_metrics'].loc[metrics, :]
    exp2 = exp2['ffc_metrics'].loc[metrics, :]
    exp3 = exp3['ffc_metrics'].loc[metrics, :]
    df_control = pd.concat(pd.DataFrame({'metric':k, 'value':v}) for k, v in control.items())
    df_control['model'] = control_id
    df_control['metric'] = df_control.index
    df_exp = pd.concat(pd.DataFrame({'metric':k, 'value':v}) for k, v in exp.items())
    df_exp['model'] = exp_id
    df_exp['metric'] = df_exp.index

    df_exp2 = pd.concat(pd.DataFrame({'metric':k, 'value':v}) for k, v in exp2.items())
    df_exp2['model'] = exp_id2
    df_exp2['metric'] = df_exp2.index
    df_exp3 = pd.concat(pd.DataFrame({'metric':k, 'value':v}) for k, v in exp3.items())
    df_exp3['model'] = exp_id3
    df_exp3['metric'] = df_exp3.index
    plot_data = df_control.append(df_exp)
    plot_data = plot_data.append(df_exp2)
    plot_data = plot_data.append(df_exp3)
    plot_data = plot_data.fillna(value=np.nan)
    plot_data['value'] = pd.to_numeric(plot_data['value'], errors='coerce')
    plot_data = plot_data.rename(columns={'value': y_label})
    # import pdb; pdb.set_trace()
    plt.figure(figsize=(7,5))
    g = sns.swarmplot(x = 'metric', y = y_label, hue = 'model', data=plot_data, palette='Set2', dodge=True, size=3)
    plt.legend(loc='upper left', labels=legend_labels)
    # g.set_ylim(0, ymax)
    # plt.show(g)
    plt.savefig('data_outputs/plots/jitter/'+filename+'.pdf')
    # put data into jitterplot with panels for each metric 
    # output and save

def line_plots_temp_precip(ffc_data):
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

def scatterplot_temp_precip(ffc_data):
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
        
        fig.savefig('data_outputs/plots/boxplots/wet_tim_mag_{}.pdf'.format(temp_sim[0]['gage_id'][0:3]))
        # plt.show()
        # import pdb; pdb.set_trace()

def scatterplot(ffc_data):
    for index, run in enumerate(ffc_data):
        if run['gage_id'] == 'run1':
            run1 = run
        if run['gage_id'] == 'run25':
            exp = run
    title = 'Extreme High'
    mag_metric = 'DS_Mag_50' # FA_Mag' , 'Wet_BFL_Mag_50' , 'SP_Mag' , DS_Mag_50
    time_metric = 'DS_Tim' # 'FA_Tim' , 'Wet_Tim' , 'SP_Tim' , DS_Tim
    dsmag_control = np.nanmean(pd.to_numeric(run1['ffc_metrics'].loc[mag_metric], errors='coerce'))
    dstim_control = np.nanmean(pd.to_numeric(run1['ffc_metrics'].loc[time_metric], errors='coerce'))

    fig, ax = plt.subplots(figsize=(8,8))
    contl_x = pd.to_numeric(run1['ffc_metrics'].loc[time_metric], errors='coerce')
    contl_y = pd.to_numeric(run1['ffc_metrics'].loc[mag_metric], errors='coerce')
    # ax.scatter(contl_x, contl_y, color='black', label='Control DT0DP1')
    # ax.set_ylim(-100, 7500) # fall: max 8600 , wet: max 7500 , sp: max 60000 , dry: 700
    # ax.set_xlim(-1, 185) # fall: (-1, 60) , wet: (-1, 185) , sp: (50, 350) , dry: (230, 395)

    x = pd.to_numeric(exp['ffc_metrics'].loc[time_metric], errors='coerce')
    y = pd.to_numeric(exp['ffc_metrics'].loc[mag_metric], errors='coerce')

    for i in range(len(contl_x)):
        plt.plot([contl_x[i],x[i]], [contl_y[i],y[i]], color='grey')
        plt.scatter(contl_x[i], contl_y[i], marker='o', color='blue', zorder=2)
        plt.scatter(x[i], y[i], marker='o', color='red', zorder=2)

    # plot scatters one more time to make marker labels
    plt.scatter(contl_x[0], contl_y[0], marker='o', color='blue', zorder=2, label='control')
    plt.scatter(x[0], y[0], marker='o', color='red', zorder=2, label='max intensity')
    plt.axhline(y=dsmag_control, ls='--', color='black', label='Average control timing')
    plt.axvline(x=dstim_control, ls=':', color='black', label='Average control magnitude')
    # import pdb; pdb.set_trace()
    
    # Attempt to order legend did not work, consider trying again later
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels, handles))
    # plt.legend(labels, handles)

    plt.legend(fancybox=True, borderaxespad = .9, fontsize='small', labelspacing=.2, columnspacing=1, markerscale=.5)
    
    plt.ylabel("Dry Season Magnitude")
    plt.xlabel("Dry Season Timing")
    plt.title(title)
    
    fig.savefig('data_outputs/plots/scatter/dry_tim_mag_intensity_shift_extremehigh.pdf')
    # plt.show()
    # import pdb; pdb.set_trace()

def line_plots(ffc_data):
    metric = 'Wet_Tim'
    #Gather model runs for line plotting
    for index, run in enumerate(ffc_data):
        if run['gage_id'] == 'run1':
            run1 = run
        if run['gage_id'] == 'run10':
            run2 = run
        if run['gage_id'] == 'run11':
            run3 = run
        if run['gage_id'] == 'run12':
            run4 = run
        if run['gage_id'] == 'run13':
            run5 = run
    runs = [run1, run2, run3, run4, run5]
    colors = ['black', '#FFC100', '#FF9800', '#FF5607', '#5C0001']
    names = ['control', 'run 10', 'run 11', 'run 12', 'run 13']
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1,1,1)
    for index, run in enumerate(runs):
        x = pd.to_numeric(run['ffc_metrics'].columns)
        y = pd.to_numeric(run['ffc_metrics'].loc[metric], errors='coerce')
        plt.plot(x, y, label=names[index], color=colors[index])
        # import pdb; pdb.set_trace()
        # plt.show()
        # plt.plot(x, control, '--', label='DT0P1_control', color='black', linewidth=.8)
        plt.title(metric)
        plt.legend(fancybox=True, borderaxespad = .9, fontsize='small', labelspacing=.2, columnspacing=1, markerscale=.5)
        fig.savefig('data_outputs/plots/lines/runs10-13_{}.pdf'.format(metric))
    # import pdb; pdb.set_trace()

# ## Bar plot for MK results, all FF metrics

# df = pd.read_csv('data_outputs/All_MK_results_RCP8.5.csv')
# df = df.set_index('Year')
# df = df.drop(['Peak_5', 'Peak_10', 'DS_No_Flow', 'Std'])
# # average results across all sites
# metric_avg = []
# for row in range(len(df)):
#    metric_avg.append(np.nanmean(df.iloc[row]))
# y = metric_avg
# x = df.index
# # Prep gini data for plotting
# gini_85 = [0.865373961,0.63434903,0.970083102,0.376731302,0.70166205,0.420775623,0.358448753,0.527977839,0.547922438,0.867036011,0.87700831,0.886980609,0.985041551,1,0.45567867,0.346814404,0.694182825,0.330193906,0.418282548,0.461495845,0.454847645,0.411634349,0.822160665,0.597783934]
# gini_45 = [0.913573407,0.865373961,0.985041551,0.554570637,0.798891967,0.682548476,0.28199446,0.537950139,0.601108033,0.920221607,0.985041551,0.985041551,1,1,0.466481994,0.311911357,0.797229917,0.466481994,0.413296399,0.530470914,0.341828255,0.306925208,0.870360111,0.675900277]

# def make_gini_stars_rank(gini_nums):
#     gini_stars = []
#     for index in range(len(gini_nums)):
#         if gini_nums[index] < 0.25:
#             gini_stars.append('')
#             continue
#         elif gini_nums[index] < 0.5:
#             gini_stars.append('*')
#             continue
#         elif gini_nums[index] < 0.75:
#             gini_stars.append('**')
#             continue
#         else:
#             gini_stars.append('***')
#     return(gini_stars)
# gini_85_stars = make_gini_stars_rank(gini_85)
# gini_45_stars = make_gini_stars_rank(gini_45)
# # create bar plot
# colors_ls = ['gold', 'gold', 'gold', 'cornflowerblue', 'cornflowerblue','cornflowerblue','cornflowerblue', 'navy', 'navy', 'navy', 'navy', 'navy', 'navy', 'navy',
# 'yellowgreen', 'yellowgreen','yellowgreen','yellowgreen','lightcoral','lightcoral','lightcoral','lightcoral','grey','grey']
# bars = plt.bar(x, y,  color = colors_ls, edgecolor='black', linewidth=1)
# plt.axis((None,None,-10,10))
# plt.margins(x=.75)

# for index, bar in enumerate(bars):
#     height = bar.get_height()
#     if height >=0:
#         bar_height = height - 0.2
#     elif height < 0: 
#         bar_height = height - .85
#     # import pdb; pdb.set_trace()
#     plt.text(bar.get_x() + bar.get_width()/2., bar_height, gini_85_stars[index],
#         ha='center', va='bottom')

# plt.xticks(rotation = 290) 
# plt.xticks(fontsize= 8)
# plt.tight_layout()
# plt.savefig('mk_trends_gini_stars.pdf')
# # import pdb; pdb.set_trace()
# # specify colors (according to ff  comp)
# # title: RCP 8.5/4.5
# # output as png