
def test():
    print('test2')

# function to fit a 2D Gaussian mixture model to the 'sat' data variable of an xarray dataset    

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore
import numpy as np
from sklearn.mixture import GaussianMixture
import xarray as xr
import matplotlib.pyplot as plt

colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

def fit_gmm(ds_sat, n_pc=2, n_components=3):
    """Fit a 2D Gaussian mixture model to the 'sat' data variable of an xarray dataset.

    Parameters
    ----------
    ds_sat : xarray.Dataset
        An xarray dataset containing a 'sat' data variable with dimensions (age, lat, lon).
    n_pc : int
        Number of principal components to retain after PCA
    n_components : int
        Number of components in the Gaussian mixture model

    Returns
    -------
    xarray.Dataset
        The updated xarray dataset with a new 'class_label' data variable.

    """
    if n_components > 10:
        raise ValueError('n_components cannot be greater than 10, otherwise the colors will repeat.')

    # get all the sat from ds_sat and put it to a ndarray
    sat = ds_sat['sat'].values
    sat_shape = sat.shape

    # reshape the sat to a 2D array
    sat = sat.reshape(sat_shape[0], sat_shape[1]*sat_shape[2])
    sat = sat.T

    # Normalize the data
    scaler = StandardScaler()
    sat_scaled = scaler.fit_transform(sat)

    # PCA
    pca_sat = PCA(n_components=n_pc)
    columns = [f'pc {i}' for i in range(1, n_pc+1)]
    PCs = pca_sat.fit_transform(sat_scaled.T)

    # calculate the PCA scores for the original unscaled data
    sat_scores = sat_scaled.dot(zscore(PCs))

    # create a 2D GMM model
    gmm_model = GaussianMixture(n_components=n_components, covariance_type='full')

    # fit the model to the two columns of PCA scores
    gmm_model.fit(sat_scores)

    # get the predicted class labels for each data point
    class_labels = gmm_model.predict(sat_scores)

    # add the class labels to the xarray dataset
    ds_sat['class_label'] = (('lat', 'lon'), class_labels.reshape(sat_shape[1], sat_shape[2]))

    # plot the results
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=300)

    # plot the scatter plot of the two columns
    for i in range(n_components):
        mask = class_labels == i
        # ax[0].scatter(sat_scores[:, 0][mask], sat_scores[:, 1][mask], s=10, alpha=0.5, color=colors[i % len(colors)])
        ax[0].scatter(sat_scores[:, 0][mask], sat_scores[:, 1][mask], s=10, alpha=0.5, color=colors[i])

    # plot the contour plot of the fitted GMM
    x, y = np.meshgrid(np.linspace(np.min(sat_scores[:, 0]), np.max(sat_scores[:, 0]), 100),
                       np.linspace(np.min(sat_scores[:, 1]), np.max(sat_scores[:, 1]), 100))
    XX = np.array([x.ravel(), y.ravel()]).T
    Z = -gmm_model.score_samples(XX)
    Z = Z.reshape(x.shape)
    ax[1].contour(x, y, Z, cmap='coolwarm_r')

    # Add labels and title
    ax[0].set_xlabel('PC 1')
    ax[0].set_ylabel('PC 2')
    ax[1].set_xlabel('PC 1')
    ax[1].set_ylabel('PC 2')
    ax[0].set_title('Scatter plot of PCA scores')
    ax[1].set_title('Contour plot of fitted GMM')

    plt.tight_layout()
    plt.show()

    #################################################################################################

# function to plot the class labels on a map

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def plot_labels(ds):
    sat_label = ds['class_label']

    # create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(projection=ccrs.PlateCarree()),dpi=300)

    # add coastline and gridlines
    ax.add_feature(cfeature.COASTLINE)
    ax.gridlines()

    # plot heatmap
    # cmap = plt.cm.get_cmap('tab20b', len(np.unique(sat_label)))
    # create colormap with unique colors for each class label
    # colors = list(mcolors.CSS4_COLORS.values())
    # colors = list(mcolors.TABLEAU_COLORS.values())
    # print(colors[1:len(np.unique(sat_label))+1])
    # cmap = mcolors.ListedColormap(colors[1:len(np.unique(sat_label))+1])
    cmap = mcolors.ListedColormap(colors[0:len(np.unique(sat_label))])

    im = ax.pcolormesh(ds.lon, ds.lat, sat_label, transform=ccrs.PlateCarree(), cmap=cmap, shading='auto')
    
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    # add lon and lat labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spatial distribution of class labels')

    # add colorbar
    bounds = np.arange(len(np.unique(sat_label))+1) -0.5
    ticks = np.arange(len(np.unique(sat_label)))
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, boundaries=bounds, ticks=ticks)
    cbar.ax.set_yticklabels(np.unique(sat_label))
    cbar.ax.set_ylabel('Class Label')

    # set title and show plot
    plt.show()

##############################################################################################
# find the deglacial sequence of each class (label) 

import numpy as np
import xarray as xr


# label2plot=[]
def find_labels_transition(ds):
    min_age_list=[]
    labels=[]
    classlabel_age=[]
    sequence_class_age=[]

    ds['transition_age_of_label'] = xr.full_like(ds['sat'].isel(age=0), np.nan)
    for label in np.unique(ds['class_label']):
        # label2plot = label
        label_mask = ds['class_label'] == label
        label_sat = ds['sat'].where(label_mask)
        label_sat_average = label_sat.mean(dim=('lat', 'lon'))
        min_age = ds['age'].isel(age=label_sat_average.argmin(dim='age'))
        # min_sat = label_sat_average.min(dim='age')
        min_age_list.append(min_age.values)
        labels.append(label)
        classlabel_age.append([label,min_age.values])
        age_min_sat = xr.where(label_mask, min_age, np.nan)
        ds['transition_age_of_label'] = ds['transition_age_of_label'].where(~label_mask, age_min_sat)

    # compute sequence
    classlabel_age.sort(key=lambda x: x[1], reverse=True)
    ds['sequence'] = xr.full_like(ds['sat'].isel(age=0), np.nan)
    #min_age_list = np.unique(ds['transition_age_of_label'].values[~np.isnan(ds['transition_age_of_label'].values)])
    # min_age_list.sort()
    # print(min_age_list)
    for i in range(len(classlabel_age)):#enumerate(np.unique(ds['class_label'])):
        label_mask = ds['class_label'] == classlabel_age[i][0]
        sequence_class_age.append([i,classlabel_age[i][0],classlabel_age[i][1]])
        ds['sequence'] = xr.where(label_mask, i, ds['sequence'])
    ds['sequence'] = ds['sequence'].fillna(-2147483648).astype(int)


    # ncol=2
    # nrow=int(np.ceil(len(np.unique(ds['class_label']))/2))
    # fig,ax=plt.subplots(nrow,ncol,figsize=(16,8*nrow))

    # for i in range(len(classlabel_age)):
    #         # get the class label based on the sequence
    #         label2plot = classlabel_age[i][0]
    #         label_mask = ds['class_label'] == label2plot
    #         label_sat = ds['sat'].where(label_mask)
    #         label_sat_average = label_sat.mean(dim=('lat', 'lon'))
    #         min_age = ds['age'].isel(age=label_sat_average.argmin(dim='age'))
    #         min_sat = label_sat_average.min(dim='age')
    #         row=int(i/ncol)
    #         col=i%ncol
    #         ax[row,col].plot(ds.age,label_sat_average,color='red')
    #         ax[row,col].plot(min_age,min_sat,'bo')
    #         ax[row,col].set_xlabel('Age')
    #         ax[row,col].set_ylabel('Sat')
    #         ax[row,col].set_title(f'Class Label: {str(label2plot)}, Sequence: {str(i)}, Age of transition: {str(min_age.values)}')
    # plt.show()

    return ds,sequence_class_age

##############################################################################################
# function to plot the class labels on a map

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np


def plot_all_label_at_sequence(ds,plot_class=False):
    ds,sequence_class_age=find_labels_transition(ds)
    sequence_label = ds['sequence']
    sat_label = ds['class_label']
    # create a figure and axis
    nrow=len(np.unique(sat_label))
    fig = plt.figure(figsize=(16, 5*nrow),dpi=300)

    for i in np.unique(ds['sequence']):
        label2plot=sequence_class_age[i][1]
        # add coastline and gridlines
        ax = fig.add_subplot(nrow, 2, 2*i+1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.gridlines()

        # plot heatmap
        # colors = list(mcolors.TABLEAU_COLORS.values())
        # colors = colors[1:len(np.unique(sat_label))+1]
        cmap = mcolors.ListedColormap(['#ffffff', colors[label2plot]])
        im = ax.pcolormesh(ds.lon, ds.lat, sat_label==label2plot, transform=ccrs.PlateCarree(), cmap=cmap, shading='auto')   
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
        # add lon and lat labels
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Sequence Label: {str(sequence_class_age[i][0])}, Class label: {str(sequence_class_age[i][1])}, Age of transition: {str(int(sequence_class_age[i][2]))}')


    ##########################################
    # calculate average sat data for each label

        label_mask = ds['class_label'] == label2plot
        label_sat = ds['sat'].where(label_mask)
        label_sat_average = label_sat.mean(dim=('lat', 'lon'))
        min_age = ds['age'].isel(age=label_sat_average.argmin(dim='age'))
        min_sat = label_sat_average.min(dim='age')

        # create axis for the second plot
        ax1 = fig.add_subplot(nrow, 2, 2*i+2)

        # timing for climate transitions, data from Rasmussen et al., 2014, in b2k
        HS1=np.array([17480,14692])-50 # convert to b1950
        BA=np.array([14692,12896])-50
        YD=np.array([12896,11703])-50

        # plot the timing of climate transitions using vertical lines
        ax1.axvline(x=HS1[0],color='black',linestyle='--') # HS1
        ax1.axvline(x=HS1[1],color='black',linestyle='--') # HS1

        ax1.axvline(x=BA[0],color='black',linestyle='--') # BA
        ax1.axvline(x=BA[1],color='black',linestyle='--') # BA

        ax1.axvline(x=YD[0],color='black',linestyle='--') # YD
        ax1.axvline(x=YD[1],color='black',linestyle='--') # YD

        if plot_class:
            # plot all sat data in the same label as light grey lines
            # reshape the label_sat to a 2D array
            label_sat = label_sat.values
            label_sat = label_sat.reshape(label_sat.shape[0], label_sat.shape[1]*label_sat.shape[2])
            print(label_sat.shape)
            for i in range(label_sat.shape[1]):
                ax1.plot(ds['age'],label_sat[:,i], color='lightgray', alpha=0.1)

        ax1.plot(ds.age,label_sat_average,color=colors[label2plot])
        ax1.plot(min_age,min_sat,'ko')
        # print(np.max(ax1.get_ylim()))
        
        # add labels for the vertical lines
        ax1.text(HS1[0]-0.7*(HS1[0]-HS1[1]),np.max(ax1.get_ylim())-0.1*(np.max(ax1.get_ylim())-np.min(ax1.get_ylim())),'HS1',rotation=90)
        ax1.text(BA[0]-0.7*(BA[0]-BA[1]),np.max(ax1.get_ylim())-0.1*(np.max(ax1.get_ylim())-np.min(ax1.get_ylim())),'BA',rotation=90)
        ax1.text(YD[0]-0.7*(YD[0]-YD[1]),np.max(ax1.get_ylim())-0.1*(np.max(ax1.get_ylim())-np.min(ax1.get_ylim())),'YD',rotation=90)

        ax1.set_xlabel('Age')
        ax1.set_ylabel('Sat')
        #ax1.set_title('Class Label: '+str(label2plot))

    plt.show()










#####################################################################################################

# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from scipy.stats import zscore
# import numpy as np
# from sklearn.mixture import GaussianMixture
# import xarray as xr
# import matplotlib.pyplot as plt

# colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# def fit_gmm(ds_sat,T_var_name='sat',n_pc=2, n_components=3):
#     """Fit a 2D Gaussian mixture model to the 'sat' data variable of an xarray dataset.

#     Parameters
#     ----------
#     ds_sat : xarray.Dataset
#         An xarray dataset containing a 'sat' data variable with dimensions (age, lat, lon).
#     n_pc : int
#         Number of principal components to retain after PCA
#     n_components : int
#         Number of components in the Gaussian mixture model

#     Returns
#     -------
#     xarray.Dataset
#         The updated xarray dataset with a new 'class_label' data variable.

#     """
#     if n_components > 10:
#         raise ValueError('n_components cannot be greater than 10, otherwise the colors will repeat.')

#     # get all the sat from ds_sat and put it to a ndarray
#     sat = ds_sat[T_var_name].values
#     sat_shape = sat.shape

#     # reshape the sat to a 2D array
#     sat = sat.reshape(sat_shape[0], sat_shape[1]*sat_shape[2])
#     sat = sat.T

#     # Normalize the data
#     scaler = StandardScaler()
#     sat_scaled = scaler.fit_transform(sat)

#     # PCA
#     pca_sat = PCA(n_components=n_pc)
#     columns = [f'pc {i}' for i in range(1, n_pc+1)]
#     PCs = pca_sat.fit_transform(sat_scaled.T)

#     # calculate the PCA scores for the original unscaled data
#     sat_scores = sat_scaled.dot(zscore(PCs))

#     # create a 2D GMM model
#     gmm_model = GaussianMixture(n_components=n_components, covariance_type='full')

#     # fit the model to the two columns of PCA scores
#     gmm_model.fit(sat_scores)

#     # get the predicted class labels for each data point
#     class_labels = gmm_model.predict(sat_scores)

#     # add the class labels to the xarray dataset
#     ds_sat['class_label'] = (('lat', 'lon'), class_labels.reshape(sat_shape[1], sat_shape[2]))

#     # plot the results
#     fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=300)

#     # plot the scatter plot of the two columns
#     for i in range(n_components):
#         mask = class_labels == i
#         # ax[0].scatter(sat_scores[:, 0][mask], sat_scores[:, 1][mask], s=10, alpha=0.5, color=colors[i % len(colors)])
#         ax[0].scatter(sat_scores[:, 0][mask], sat_scores[:, 1][mask], s=10, alpha=0.5, color=colors[i])

#     # plot the contour plot of the fitted GMM
#     x, y = np.meshgrid(np.linspace(np.min(sat_scores[:, 0]), np.max(sat_scores[:, 0]), 100),
#                        np.linspace(np.min(sat_scores[:, 1]), np.max(sat_scores[:, 1]), 100))
#     XX = np.array([x.ravel(), y.ravel()]).T
#     Z = -gmm_model.score_samples(XX)
#     Z = Z.reshape(x.shape)
#     ax[1].contour(x, y, Z, cmap='coolwarm_r')

#     # Add labels and title
#     ax[0].set_xlabel('PC 1')
#     ax[0].set_ylabel('PC 2')
#     ax[1].set_xlabel('PC 1')
#     ax[1].set_ylabel('PC 2')
#     ax[0].set_title('Scatter plot of PCA scores')
#     ax[1].set_title('Contour plot of fitted GMM')

#     plt.tight_layout()
#     plt.show()


# ##############################################################################################
# import xarray as xr
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import numpy as np

# def plot_labels(ds):
#     sat_label = ds['class_label']

#     # create a figure and axis
#     fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(projection=ccrs.PlateCarree()),dpi=300)

#     # add coastline and gridlines
#     ax.add_feature(cfeature.COASTLINE)
#     ax.gridlines()

#     # plot heatmap
#     # cmap = plt.cm.get_cmap('tab20b', len(np.unique(sat_label)))
#     # create colormap with unique colors for each class label
#     # colors = list(mcolors.CSS4_COLORS.values())
#     # colors = list(mcolors.TABLEAU_COLORS.values())
#     # print(colors[1:len(np.unique(sat_label))+1])
#     # cmap = mcolors.ListedColormap(colors[1:len(np.unique(sat_label))+1])
#     cmap = mcolors.ListedColormap(colors[0:len(np.unique(sat_label))])

#     im = ax.pcolormesh(ds.lon, ds.lat, sat_label, transform=ccrs.PlateCarree(), cmap=cmap, shading='auto')
    
#     ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
#     ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
#     # add lon and lat labels
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
#     ax.set_title('Spatial distribution of class labels')

#     # add colorbar
#     bounds = np.arange(len(np.unique(sat_label))+1) -0.5
#     ticks = np.arange(len(np.unique(sat_label)))
#     cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, boundaries=bounds, ticks=ticks)
#     cbar.ax.set_yticklabels(np.unique(sat_label))
#     cbar.ax.set_ylabel('Class Label')

#     # set title and show plot
#     plt.show()


# ##############################################################################################

# # find the deglacial sequence of each class (label) 
# import numpy as np
# import xarray as xr


# # label2plot=[]
# def find_labels_transition(ds,T_var_name='sat',Age_var_name='age'):
#     min_age_list=[]
#     labels=[]
#     classlabel_age=[]
#     sequence_class_age=[]

#     ds['transition_age_of_label'] = xr.full_like(ds[T_var_name].isel(age=0), np.nan)
#     for label in np.unique(ds['class_label']):
#         # label2plot = label
#         label_mask = ds['class_label'] == label
#         label_sat = ds[T_var_name].where(label_mask)
#         label_sat_average = label_sat.mean(dim=('lat', 'lon'))
#         min_age = ds[Age_var_name].isel(age=label_sat_average.argmin(dim=Age_var_name))
#         # min_sat = label_sat_average.min(dim='age')
#         min_age_list.append(min_age.values)
#         labels.append(label)
#         classlabel_age.append([label,min_age.values])
#         age_min_sat = xr.where(label_mask, min_age, np.nan)
#         ds['transition_age_of_label'] = ds['transition_age_of_label'].where(~label_mask, age_min_sat)

#     # compute sequence
#     classlabel_age.sort(key=lambda x: x[1], reverse=True)
#     ds['sequence'] = xr.full_like(ds[T_var_name].isel(age=0), np.nan)
#     #min_age_list = np.unique(ds['transition_age_of_label'].values[~np.isnan(ds['transition_age_of_label'].values)])
#     # min_age_list.sort()
#     # print(min_age_list)
#     for i in range(len(classlabel_age)):#enumerate(np.unique(ds['class_label'])):
#         label_mask = ds['class_label'] == classlabel_age[i][0]
#         sequence_class_age.append([i,classlabel_age[i][0],classlabel_age[i][1]])
#         ds['sequence'] = xr.where(label_mask, i, ds['sequence'])
#     ds['sequence'] = ds['sequence'].fillna(-2147483648).astype(int)


#     # ncol=2
#     # nrow=int(np.ceil(len(np.unique(ds['class_label']))/2))
#     # fig,ax=plt.subplots(nrow,ncol,figsize=(16,8*nrow))

#     # for i in range(len(classlabel_age)):
#     #         # get the class label based on the sequence
#     #         label2plot = classlabel_age[i][0]
#     #         label_mask = ds['class_label'] == label2plot
#     #         label_sat = ds['sat'].where(label_mask)
#     #         label_sat_average = label_sat.mean(dim=('lat', 'lon'))
#     #         min_age = ds['age'].isel(age=label_sat_average.argmin(dim='age'))
#     #         min_sat = label_sat_average.min(dim='age')
#     #         row=int(i/ncol)
#     #         col=i%ncol
#     #         ax[row,col].plot(ds.age,label_sat_average,color='red')
#     #         ax[row,col].plot(min_age,min_sat,'bo')
#     #         ax[row,col].set_xlabel('Age')
#     #         ax[row,col].set_ylabel('Sat')
#     #         ax[row,col].set_title(f'Class Label: {str(label2plot)}, Sequence: {str(i)}, Age of transition: {str(min_age.values)}')
#     # plt.show()

#     return ds,sequence_class_age


# ##############################################################################################

# import xarray as xr
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import numpy as np

# import GMM_functions as gf

# def plot_all_label_at_sequence(ds,T_var_name='sat',Age_var_name='age',plot_class=False):
#     ds,sequence_class_age=gf.find_labels_transition(ds)
#     sequence_label = ds['sequence']
#     sat_label = ds['class_label']
#     # create a figure and axis
#     nrow=len(np.unique(sat_label))
#     fig = plt.figure(figsize=(16, 5*nrow),dpi=300)

#     for i in np.unique(ds['sequence']):
#         label2plot=sequence_class_age[i][1]
#         # add coastline and gridlines
#         ax = fig.add_subplot(nrow, 2, 2*i+1, projection=ccrs.PlateCarree())
#         ax.add_feature(cfeature.COASTLINE)
#         ax.gridlines()

#         # plot heatmap
#         # colors = list(mcolors.TABLEAU_COLORS.values())
#         # colors = colors[1:len(np.unique(sat_label))+1]
#         cmap = mcolors.ListedColormap(['#ffffff', colors[label2plot]])
#         im = ax.pcolormesh(ds.lon, ds.lat, sat_label==label2plot, transform=ccrs.PlateCarree(), cmap=cmap, shading='auto')   
#         ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
#         ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
#         # add lon and lat labels
#         ax.set_xlabel('Longitude')
#         ax.set_ylabel('Latitude')
#         ax.set_title(f'Sequence Label: {str(sequence_class_age[i][0])}, Class label: {str(sequence_class_age[i][1])}, Age of transition: {str(sequence_class_age[i][2])}')


#     ##########################################
#     # calculate average sat data for each label

#         label_mask = ds['class_label'] == label2plot
#         label_sat = ds[T_var_name].where(label_mask)
#         label_sat_average = label_sat.mean(dim=('lat', 'lon'))
#         min_age = ds[Age_var_name].isel(age=label_sat_average.argmin(dim=Age_var_name))
#         min_sat = label_sat_average.min(dim=Age_var_name)

#         # create axis for the second plot
#         ax1 = fig.add_subplot(nrow, 2, 2*i+2)

#         # timing for climate transitions, data from Rasmussen et al., 2014, in b2k
#         HS1=np.array([17480,14692])-50 # convert to b1950
#         BA=np.array([14692,12896])-50
#         YD=np.array([12896,11703])-50

#         # plot the timing of climate transitions using vertical lines
#         ax1.axvline(x=HS1[0],color='black',linestyle='--') # HS1
#         ax1.axvline(x=HS1[1],color='black',linestyle='--') # HS1

#         ax1.axvline(x=BA[0],color='black',linestyle='--') # BA
#         ax1.axvline(x=BA[1],color='black',linestyle='--') # BA

#         ax1.axvline(x=YD[0],color='black',linestyle='--') # YD
#         ax1.axvline(x=YD[1],color='black',linestyle='--') # YD

#         if plot_class:
#             # plot all sat data in the same label as light grey lines
#             # reshape the label_sat to a 2D array
#             label_sat = label_sat.values
#             label_sat = label_sat.reshape(label_sat.shape[0], label_sat.shape[1]*label_sat.shape[2])
#             print(label_sat.shape)
#             for i in range(label_sat.shape[1]):
#                 ax1.plot(ds[Age_var_name],label_sat[:,i], color='lightgray', alpha=0.1)

#         ax1.plot(ds[Age_var_name],label_sat_average,color=colors[label2plot])
#         ax1.plot(min_age,min_sat,'ko')
#         # print(np.max(ax1.get_ylim()))
        
#         # add labels for the vertical lines
#         ax1.text(HS1[0]-0.7*(HS1[0]-HS1[1]),np.max(ax1.get_ylim())-0.1*(np.max(ax1.get_ylim())-np.min(ax1.get_ylim())),'HS1',rotation=90)
#         ax1.text(BA[0]-0.7*(BA[0]-BA[1]),np.max(ax1.get_ylim())-0.1*(np.max(ax1.get_ylim())-np.min(ax1.get_ylim())),'BA',rotation=90)
#         ax1.text(YD[0]-0.7*(YD[0]-YD[1]),np.max(ax1.get_ylim())-0.1*(np.max(ax1.get_ylim())-np.min(ax1.get_ylim())),'YD',rotation=90)

#         ax1.set_xlabel('Age')
#         ax1.set_ylabel('Sat')
#         #ax1.set_title('Class Label: '+str(label2plot))

#     plt.show()
