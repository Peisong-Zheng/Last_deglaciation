# import numpy as np
# import Rbeast as rb
# import matplotlib.pyplot as plt

# def find_cp(data, age,interval_L_indx=10,rb_plot=False,avg_plot=False,avg_plot_title=None):

#     # cut the data according to interval_L_indx
#     data=data[interval_L_indx:]
#     age=age[interval_L_indx:]



#     # flip the data
#     data = data[::-1]

#     # start_age = ds_sat_EOFs['age'][-1].values
#     start_age=age[-1]
#     # print(start_age)

#     o = rb.beast(data, start=0, season='none')
#     if rb_plot:
#         rb.plot(o)

#     cps = o.trend.cp
#     # remove nan
#     cps = cps[~np.isnan(cps)]

#     slpSgnPosPr_list = [o.trend.slpSgnPosPr[int(cp)] for cp in cps]
#     slpSgnZeroPr_list = [o.trend.slpSgnZeroPr[int(cp)] for cp in cps]
#     slpSgnNegPr_list = 1 - (np.array(slpSgnPosPr_list) + np.array(slpSgnZeroPr_list))

#     # calculate the 5th percentile of the slpSgnPosPr_list
#     slpSgnPosPr_50th = np.percentile(slpSgnPosPr_list, 50)
#     # calculate the 5th percentile of the slpSgnNegPr_list
#     slpSgnNegPr_50th = np.percentile(slpSgnNegPr_list, 50)

#     cp_age_list = [start_age - cp*200 for cp in cps]

#     flag = 'None'

#     # Checking the conditions for the change points
#     selected_cp_index = None
#     for i, (pos_pr, neg_pr) in enumerate(zip(slpSgnPosPr_list, slpSgnNegPr_list)):
#         print('pos_pr',pos_pr)
#         print('neg_pr',neg_pr)
#         if pos_pr <= 0.5 and neg_pr >= 0.2:
#             selected_cp_index = i
#             flag = 'slope'
#             break
#         # if pos_pr != 1 and neg_pr != 0:
#         #     selected_cp_index = i
#         #     flag = 'slope'
#         #     break

#     # If no change point satisfies the condition, get the change point with the largest age
#     if selected_cp_index is None:
#         selected_cp_index = np.argmax(cp_age_list)
#         flag = 'maxage'

#     selected_cp_age = cp_age_list[selected_cp_index]
#     # value_at_cp = data[int(cps[selected_cp_index])]

#     ##########################################################
#     # data=data[interval_L_indx:]
#     # age=age[interval_L_indx:]

#     # # flip the data
#     # data = data[::-1]

#     # # start_age = ds_sat_EOFs['age'][-1].values
#     # start_age=age[-1]
#     # # print(start_age)

#     # o = rb.beast(data, start=0, season='none')

#     # rb.plot(o)

#     # cps = o.trend.cp
#     # # remove nan
#     # cps = cps[~np.isnan(cps)]
#     # # print('cps',cps)

#     # slpSgnPosPr_list = [o.trend.slpSgnPosPr[int(cp)] for cp in cps]
#     # # slpSgnZeroPr_list = [o.trend.slpSgnZeroPr[int(cp)] for cp in cps]
#     # slpSgnNegPr_list = 1 - np.array(slpSgnPosPr_list) 

#     # cp_age_list = [start_age - cp*200 for cp in cps]

#     # # print('cp_age_list',cp_age_list)
#     # # print('slpSgnPosPr',o.trend.slpSgnPosPr)
#     # # print('slpSgnPosPr_list',slpSgnPosPr_list)


#     # sorted_indices = np.argsort(cp_age_list)[::-1]

#     # sorted_cp_age_list = np.array(cp_age_list)[sorted_indices]#.tolist()
#     # sorted_slpSgnPosPr_list = np.array(slpSgnPosPr_list)[sorted_indices]#.tolist()
#     # # sorted_slpSgnZeroPr_list = np.array(slpSgnZeroPr_list)[sorted_indices].tolist()
#     # sorted_slpSgnNegPr_list = np.array(slpSgnNegPr_list)[sorted_indices]#.tolist()

#     # # print('sorted_cp_age_list',sorted_cp_age_list)
#     # # print('sorted_slpSgnPosPr_list',sorted_slpSgnPosPr_list)


#     # flag = 'None'

#     # # Checking the conditions for the change points
#     # selected_cp_index = None
#     # for i, (pos_pr, neg_pr) in enumerate(zip(sorted_slpSgnPosPr_list, sorted_slpSgnNegPr_list)):
#     #     if pos_pr != 1 and neg_pr != 0:
#     #         selected_cp_index = i
#     #         flag = 'slope'
#     #         break

#     # # If no change point satisfies the condition, get the change point with the largest age
#     # if selected_cp_index is None:
#     #     selected_cp_index = np.argmax(sorted_cp_age_list)
#     #     flag = 'maxage'

#     # selected_cp_age = sorted_cp_age_list[selected_cp_index]
#     # print('selected_cp_age',selected_cp_age)

#     ###############

#     cpOccPr = o.trend.cpOccPr    
#     # flip the cpOccPr
#     cpOccPr = cpOccPr[::-1]
    
#     slpSgnPosPr = 1 - o.trend.slpSgnPosPr
#     slpSgnZeroPr = o.trend.slpSgnZeroPr
#     # flip the slpSgnPosPr
#     slpSgnPosPr = slpSgnPosPr[::-1]
#     slpSgnZeroPr = slpSgnZeroPr[::-1]


#     print('flag:', flag)

#     if avg_plot:
#         # plot the data and age time series
#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.plot(age, data[::-1], color='k', label='data')
#         ax.axvline(selected_cp_age, color='r', label='change point')
#         # add text to label the cp age
#         ax.text(selected_cp_age, np.min(ax.get_ylim()), str(int(selected_cp_age)), fontsize=12, color='r')
#         ax.set_xlabel('Age (yr BP)')
#         ax.set_ylabel('Weighted average SAT (°C)')
#         ax.set_title(avg_plot_title)
#         # invert the x axis
#         ax.invert_xaxis()


#     output = {
#     'data_flipped': data[::-1],
#     'cpOccPr': cpOccPr,
#     'slpSgnPosPr': slpSgnPosPr,
#     'slpSgnZeroPr': slpSgnZeroPr,
#     'cp_age': selected_cp_age,
#     'age': age,
#     }

#     return output


##############################################################################################
# import numpy as np
# import Rbeast as rb
# import matplotlib.pyplot as plt

# def find_cp(data, age,interval_L_indx=10,rb_plot=False,avg_plot=False,avg_plot_title=None):

#     # cut the data according to interval_L_indx
#     data=data[interval_L_indx:]
#     age=age[interval_L_indx:]



#     # flip the data
#     data = data[::-1]

#     # start_age = ds_sat_EOFs['age'][-1].values
#     start_age=age[-1]
#     # print(start_age)

#     o = rb.beast(data, start=0, season='none')
#     if rb_plot:
#         rb.plot(o)
    

#     cps = o.trend.cp
#     # remove nan
#     cps = cps[~np.isnan(cps)]

#     slpSgnPosPr_list = [o.trend.slpSgnPosPr[int(cp)] for cp in cps]
#     slpSgnZeroPr_list = [o.trend.slpSgnZeroPr[int(cp)] for cp in cps]
#     slpSgnNegPr_list = 1 - (np.array(slpSgnPosPr_list))

#     # calculate the 5th percentile of the slpSgnPosPr_list
#     slpSgnPosPr_50th = np.percentile(slpSgnPosPr_list, 50)
#     # calculate the 5th percentile of the slpSgnNegPr_list
#     slpSgnNegPr_50th = np.percentile(slpSgnNegPr_list, 50)

#     cp_age_list = [start_age - cp*200 for cp in cps]

#     flag = 'None'

#     # Checking the conditions for the change points
#     selected_cp_index = None
#     for i, (pos_pr, neg_pr) in enumerate(zip(slpSgnPosPr_list, slpSgnNegPr_list)):
#         print('pos_pr',pos_pr)
#         print('neg_pr',neg_pr)
#         if (1-pos_pr)>0.8 and neg_pr>0.5: #pos!=1 and neg>=0.05
#             selected_cp_index = i
#             flag = 'slope'
#             break

#     # If no change point satisfies the condition, get the change point with the largest age
#     if selected_cp_index is None:
#         selected_cp_index = np.argmax(cp_age_list)
#         flag = 'maxage'

#     selected_cp_age = cp_age_list[selected_cp_index]


#     cpOccPr = o.trend.cpOccPr    
#     # flip the cpOccPr
#     cpOccPr = cpOccPr[::-1]
    
#     slpSgnPosPr = 1 - o.trend.slpSgnPosPr
#     slpSgnZeroPr = o.trend.slpSgnZeroPr
#     # flip the slpSgnPosPr
#     slpSgnPosPr = slpSgnPosPr[::-1]
#     slpSgnZeroPr = slpSgnZeroPr[::-1]


#     print('flag:', flag)

#     if avg_plot:
#         # plot the data and age time series
#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.plot(age, data[::-1], color='k', label='data')
#         ax.axvline(selected_cp_age, color='r', label='change point')
#         # add text to label the cp age
#         ax.text(selected_cp_age, np.min(ax.get_ylim()), str(int(selected_cp_age)), fontsize=12, color='r')
#         ax.set_xlabel('Age (yr BP)')
#         ax.set_ylabel('Weighted average SAT (°C)')
#         ax.set_title(avg_plot_title)
#         # invert the x axis
#         ax.invert_xaxis()


#     output = {
#     'data_flipped': data[::-1],
#     'cpOccPr': cpOccPr,
#     'slpSgnPosPr': slpSgnPosPr,
#     'slpSgnZeroPr': slpSgnZeroPr,
#     'cp_age': selected_cp_age,
#     'age': age,
#     }

#     return output
import numpy as np
import Rbeast as rb
import matplotlib.pyplot as plt

def find_cp(data, age,age_step=200, interval_L_indx=10,rb_plot=False,avg_plot=False,avg_plot_title=None):

    # cut the data according to interval_L_indx
    data=data[interval_L_indx:]
    age=age[interval_L_indx:]


    # flip the data
    data = data[::-1]

    # start_age = ds_sat_EOFs['age'][-1].values
    start_age=age[-1]
    # print(start_age)

    o = rb.beast(data, start=0, season='none', options=0, quiet=1)
    if rb_plot:
        rb.plot(o)
    

    cps = o.trend.cp

    # sort the cpCI according to the cps
    cpCI=o.trend.cpCI
    cpCI_cps_stack=np.column_stack((cpCI,cps))
    # sort the cpCI_cps_stack according to the last column in asending order
    cpCI_cps_stack_sorted=cpCI_cps_stack[cpCI_cps_stack[:,1].argsort()]

    # remove nan
    cps = cps[~np.isnan(cps)]
    # sort the cps
    cps = np.sort(cps)

    pospr_diff_max_index=np.argmax(np.diff(o.trend.slpSgnPosPr))
    print('pospr_diff_max_index:', pospr_diff_max_index)
    print('cps', cps)

    # find the closet cp to the pospr_diff_max_index
    selected_cp_index = np.argmin(np.abs(cps-pospr_diff_max_index))
    print('selected_cp_index:', selected_cp_index)

    selected_cp_age=start_age - cps[selected_cp_index]*age_step
    selected_cp_age_CI=start_age-cpCI_cps_stack_sorted[selected_cp_index]*age_step
    # get the first two data in selected_cp_age_CI
    selected_cp_age_CI=selected_cp_age_CI[:2]

    # cp_CI_all = start_age-cpCI_cps_stack_sorted[cps]*age_step
    # cp_age_all = [start_age - cp*200 for cp in cps]
    

    # get the CPs with the highest(top 3) cpOccPr
    cpOccPr = o.trend.cpOccPr    

    cpOccPr_at_cps=[cpOccPr[int(cp)] for cp in cps]
    # put cpOccPr_at_cps and cps into a array with two columns
    cpOccPr_cps_stack=np.column_stack((cpOccPr_at_cps,cps))
    # in cpOccPr_cps_stack, sort the first column in descending order and change the order of the second column accordingly
    cpOccPr_cps_stack_sorted=cpOccPr_cps_stack[cpOccPr_cps_stack[:,0].argsort()[::-1]]
    # get the first 3 of the second column
    cps_top3=cpOccPr_cps_stack_sorted[:3,1]

    cp_age_all = [start_age - cp*age_step for cp in cps_top3]




    # flip the cpOccPr
    cpOccPr = cpOccPr[::-1]
    
    slpSgnPosPr = o.trend.slpSgnPosPr
    slpSgnZeroPr = o.trend.slpSgnZeroPr
    # flip the slpSgnPosPr
    slpSgnPosPr = slpSgnPosPr[::-1]
    slpSgnZeroPr = slpSgnZeroPr[::-1]


    if avg_plot:
        # plot the data and age time series
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(age, data[::-1], color='k', label='data')
        ax.axvline(selected_cp_age, color='r', label='change point')
        # add text to label the cp age
        ax.text(selected_cp_age, np.min(ax.get_ylim()), str(int(selected_cp_age)), fontsize=12, color='r')
        ax.set_xlabel('Age (yr BP)')
        ax.set_ylabel('Weighted average SAT (°C)')
        ax.set_title(avg_plot_title)
        # invert the x axis
        ax.invert_xaxis()


    output = {
    'data_flipped': data[::-1],
    'cpOccPr': cpOccPr,
    'slpSgnPosPr': slpSgnPosPr,
    'slpSgnZeroPr': slpSgnZeroPr,
    'cp_age': selected_cp_age,
    'cp_age_CI': selected_cp_age_CI,
    'cp_age_all': cp_age_all,
    'cp_CI_all': cpCI_cps_stack_sorted,
    'age': age,
    }

    return output
################################################################################################
def sort_classes_by_cp_age(unsorted_cp):
    """
    Relabels the classes based on cp_age in descending order and provides a one-to-one match between unsorted and sorted classes.

    Parameters:
    unsorted_cp (dict): A dictionary with class labels as keys and cp_age as values.

    Returns:
    tuple of two dicts:
        - The first dictionary has new class labels as keys (starting from 0 for the largest cp_age) and cp_age as values.
        - The second dictionary shows the one-to-one match between the unsorted classes and the sorted classes.
    """
    sorted_classes = sorted(unsorted_cp, key=unsorted_cp.get, reverse=True)
    relabeled_cp = {new_label: unsorted_cp[original_class] for new_label, original_class in enumerate(sorted_classes)}
    match = {original_class: new_label for new_label, original_class in enumerate(sorted_classes)}
    return relabeled_cp, match

################################################################################################
import xarray as xr

def cal_weighted_average_curve(ds, class_label,sat_var_name='sat',class_label_name='class_label'):
    
    ds_sat_subset = ds.where(ds[class_label_name] == class_label, drop=True)


    weights_broadcasted = ds_sat_subset['weight'].broadcast_like(ds_sat_subset[sat_var_name])
    sum_weighted_sat = (ds_sat_subset[sat_var_name] * weights_broadcasted).sum(dim=['lat', 'lon'])
    sum_weight_sat = weights_broadcasted.sum(dim=['lat', 'lon'])

    weighted_avg_sat = sum_weighted_sat / sum_weight_sat

    return weighted_avg_sat.values


def cal_anomalies(ds_sat, years):
    """calculate the temperature anomalies by subtracting the mean over the specified years from the 'sat' data variable of an xarray dataset.

    Parameters
    ----------
    ds_sat : xarray.Dataset
        An xarray dataset containing a 'sat' data variable with dimensions (age, lat, lon).
    years : int
        The number of years to calculate the mean over.

    Returns
    -------
    xarray.Dataset
        The updated xarray dataset with a new 'sat_anomalies' data variable.
    """
    # Select the specified years of the record
    ds_years = ds_sat.sel(age=slice(None, np.min(ds_sat['age'].values)+years))

    # Calculate the mean over the specified years
    mean_years = ds_years['sat'].mean(dim='age')

    # Compute the anomalies by subtracting the mean from the 'sat' variable
    anomalies_sat = ds_sat['sat'] - mean_years

    # Add the anomalies as a new data variable in the dataset
    ds_sat['sat_anomalies'] = anomalies_sat

    return ds_sat

################################################################################################


























# import numpy as np
# import Rbeast as rb

# def find_cp(data, age,method='b',interval_L_indx=10, n_order_left=2,plot=False):
#     # data = ds_sat_EOFs['sat'].isel(lat=lat, lon=lon)
    
#     # # Plot the sample
#     # data.plot()
#     # data = data.values
#     if method=='b':
#         # cut the data according to interval_L_indx
#         data=data[interval_L_indx:]
#         age=age[interval_L_indx:]
    


#         # flip the data
#         data = data[::-1]

#         # start_age = ds_sat_EOFs['age'][-1].values
#         start_age=age[-1]
#         # print(start_age)

#         o = rb.beast(data, start=0, season='none')
#         if plot:
#             rb.plot(o)

#         cps = o.trend.cp
#         cps = cps[~np.isnan(cps)]

#         cpPr = o.trend.cpPr
#         cpPr = cpPr[~np.isnan(cpPr)]

#         slpSgnPosPr_list = [o.trend.slpSgnPosPr[int(cp)] for cp in cps]
#         slpSgnZeroPr_list = [o.trend.slpSgnZeroPr[int(cp)] for cp in cps]
#         slpSgnNegPr_list = 1 - (np.array(slpSgnPosPr_list) + np.array(slpSgnZeroPr_list))

#         cp_age_list = [start_age - cp*200 for cp in cps]
#         flag = 'None'

#         # Checking the conditions for the change points
#         selected_cp_index = None
#         for i, (pos_pr, neg_pr) in enumerate(zip(slpSgnPosPr_list, slpSgnNegPr_list)):
#             if pos_pr != 1 and neg_pr != 0:
#                 selected_cp_index = i
#                 flag = 'slope'
#                 break

#         # If no change point satisfies the condition, get the change point with the largest age
#         if selected_cp_index is None:
#             selected_cp_index = np.argmax(cp_age_list)
#             flag = 'maxage'

#         selected_cp_age = cp_age_list[selected_cp_index]
#         value_at_cp = data[int(cps[selected_cp_index])]
#         print('flag:', flag)

#     if method=='min':
#         min_index = np.argmin(data)
#         # print('min_index:',min_index)
#         selected_cp_age = age[min_index]
#         value_at_cp = data[min_index]

#     return selected_cp_age,value_at_cp



# def prepare_plot_data(weighted_avg_sat, ds_sat_EOFs_MC):
#     # sample = weighted_avg_sat.values
#     sample = weighted_avg_sat
#     # flip the sample
#     sample = sample[::-1]

#     sat_start = ds_sat_EOFs_MC['age'][-1].values
#     print(sat_start)

#     o = rb.beast(sample, start=0, season='none')

#     cpOccPr = o.trend.cpOccPr    
#     # flip the cpOccPr
#     cpOccPr = cpOccPr[::-1]
    
#     slpSgnPosPr = 1 - o.trend.slpSgnPosPr
#     slpSgnZeroPr = o.trend.slpSgnZeroPr
#     # flip the slpSgnPosPr
#     slpSgnPosPr = slpSgnPosPr[::-1]
#     slpSgnZeroPr = slpSgnZeroPr[::-1]

#     cp_age, cp_sat = find_cp(sample, ds_sat_EOFs_MC['age'].values, method='b', plot=False)

#     return sample,cpOccPr, slpSgnPosPr, slpSgnZeroPr, cp_age

# ################################################################################################
# import matplotlib.pyplot as plt

# def plot_cp(data,ds_sat_EOFs_MC,dpi,curve_color,left_width,right_width,floating_plot_left,floating_plot_bottom,floating_yaxis_side,show_xlabel=False):
#     sample = data[0]
#     cpOccPr = data[1]
#     slpSgnPosPr = data[2]
#     slpSgnZeroPr = data[3]
#     cp_age = data[4]

#     # Create the main figure and axes
#     fig = plt.figure(figsize=(6, 4),dpi=dpi)
#     gs = fig.add_gridspec(2, 1, height_ratios=[2, 0.5], hspace=0)

#     # Plot weighted_avg_sat
#     ax0 = fig.add_subplot(gs[0, 0])
#     ax0.plot(ds_sat_EOFs_MC['age'].values, sample[::-1], label='weighted_avg_sat', color=curve_color,marker='o',markersize=5,markerfacecolor='white',markeredgewidth=1.5,linewidth=2)
#     ax0.xaxis.set_visible(False)
#     ax0.spines['bottom'].set_visible(False)
#     ax0.set_xlim([np.min(ds_sat_EOFs_MC['age'].values), np.max(ds_sat_EOFs_MC['age'].values)])
#     # ax0.set_ylim([-2, 6.5])
#     ylim_min, ylim_max = ax0.get_ylim()
#     ax0.plot([cp_age, cp_age], [ylim_min, ylim_max], color='black', linewidth=2)
#     # set ylim
#     ax0.set_ylim([ylim_min,ylim_max])
#     # add xlabel
#     ax0.set_xlabel('Age (ka)')
#     # add ylabel
#     ax0.set_ylabel('SAT (°C)')
#     for axis in ['top','left','right']:
#         ax0.spines[axis].set_linewidth(2)
#     # ax0.grid(True)











#     # Add a floating plot over ax0
#     # t_cutoff_left=20000
#     # t_cutoff_right=22500
#     t_cutoff_left=cp_age-left_width
#     t_cutoff_right=cp_age+right_width

#     age_subset=ds_sat_EOFs_MC['age'].values[(ds_sat_EOFs_MC['age'].values>t_cutoff_left) & (ds_sat_EOFs_MC['age'].values<t_cutoff_right)]
#     cpOccPr_subset=cpOccPr[(ds_sat_EOFs_MC['age'].values>t_cutoff_left) & (ds_sat_EOFs_MC['age'].values<t_cutoff_right)]

#     # left, width = 0.647, .2
#     # bottom, height = .66, .2
#     fraction=(t_cutoff_right-t_cutoff_left)/(ax0.get_xlim()[1] - ax0.get_xlim()[0])
#     left, width = floating_plot_left, .2
#     bottom, height = floating_plot_bottom, .25
#     floating_ax = fig.add_axes([left, bottom, width, height])
#     floating_ax.plot(age_subset, cpOccPr_subset, color='gray')
#     floating_ax.fill_between(age_subset, 0, cpOccPr_subset, facecolor='gray', alpha=0.3)
#     # get ylim
#     ylim_min, ylim_max = floating_ax.get_ylim()
#     ylim_max = np.round(ylim_max,1)
#     floating_ax.set_yticks([0, ylim_max])
#     floating_ax.set_ylim([ylim_min,ylim_max])
#     floating_ax.plot([cp_age, cp_age], [ylim_min,ylim_max], color='black', linewidth=2)

#     floating_ax.invert_yaxis()  # Invert the y-axis
#     floating_ax.set_xlim([t_cutoff_left, t_cutoff_right])
#     floating_ax.spines['right'].set_visible(False)
#     # floating_ax.spines['left'].set_visible(False)
#     floating_ax.spines['bottom'].set_visible(False)
#     floating_ax.spines['top'].set_visible(False)
#     floating_ax.spines['left'].set_linewidth(1.5)
#     if floating_yaxis_side=='right':
#         floating_ax.yaxis.tick_right()
#         floating_ax.spines['right'].set_visible(True)
#         floating_ax.spines['left'].set_visible(False)
#         floating_ax.spines['right'].set_linewidth(1.5)

        
#     # floating_ax.yaxis.set_visible(False)
#     # not show xtick labels 
#     floating_ax.xaxis.set_visible(False)
#     # not show ytick labels
#     # floating_ax.yaxis.set_visible(False)
#     # show xtick labels on the top side
#     floating_ax.xaxis.tick_top()











#     # Plot slpSgnPosPr
#     ax1 = fig.add_subplot(gs[1, 0])
#     # ax1.plot(ds_sat_EOFs_MC['age'].values, slpSgnPosPr, color='red', label='slpSgnPosPr',alpha=0.3)
#     # ax1.plot(ds_sat_EOFs_MC['age'].values, slpSgnZeroPr, color='purple', label='slpSgnZeroPr',alpha=0.3)
#     ax1.fill_between(ds_sat_EOFs_MC['age'].values, slpSgnPosPr, 0, facecolor='blue', alpha=0.4)
#     ax1.fill_between(ds_sat_EOFs_MC['age'].values, slpSgnPosPr, 1, facecolor='red', alpha=0.4)
#     # ax1.set_ylim([-0.1, 1.05])
#     ylim_min, ylim_max = ax1.get_ylim()
#     ax1.plot([cp_age, cp_age], [ylim_min,ylim_max], color='black', linewidth=2)
#     # set ylim
#     ax1.set_ylim([ylim_min,ylim_max])
#     ax1.set_yticks([0, 1])
#     ax1.yaxis.tick_right()
#     ax1.spines['top'].set_visible(False)
#     ax1.set_xlim([np.min(ds_sat_EOFs_MC['age'].values), np.max(ds_sat_EOFs_MC['age'].values)])

#     for axis in ['bottom','left','right']:
#         ax1.spines[axis].set_linewidth(2)
#     # add xlabel
#     if show_xlabel:
#         ax1.set_xlabel('Age (ka)')
#     # ax1.set_xlabel('Age (ka)')

#     # Adjust the layout
#     plt.show()


