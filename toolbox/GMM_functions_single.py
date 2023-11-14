import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore
import numpy as np
from sklearn.mixture import GaussianMixture
import xarray as xr
import matplotlib.pyplot as plt


def GMM4EOFS(data, ds_sat,n_components=4):
    # colors = ['blue', 'pink', 'green', 'orange', 'purple', 'brown', 'red', 'gray', 'olive', 'cyan']

    # n_components=5
    sat_shape=ds_sat['sat'].shape

    # create a 2D GMM model
    gmm_model = GaussianMixture(n_components=n_components, covariance_type='full')

    # fit the model to the two columns of PCA scores
    gmm_model.fit(data)

    # get the predicted class labels for each data point
    class_labels = gmm_model.predict(data)

    # new_ds=ds_sat.copy()
    # add the class labels to the xarray dataset
    ds=ds_sat.copy()
    ds['class_label'] = (('lat', 'lon'), class_labels.reshape(sat_shape[1], sat_shape[2]))

    unique_labels = np.unique(class_labels)
    cmap = plt.get_cmap('Accent', len(unique_labels))


    # plot the results
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=300)

    # plot the scatter plot of the two columns
    for i in range(n_components):
        mask = class_labels == i
        # ax[0].scatter(sat_scores[:, 0][mask], sat_scores[:, 1][mask], s=10, alpha=0.5, color=colors[i % len(colors)])
        ax[0].scatter(data[:, 0][mask], data[:, 1][mask], s=10, alpha=0.5, color=cmap(i))

    # plot the contour plot of the fitted GMM
    x, y = np.meshgrid(np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100),
                        np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100))
    XX = np.array([x.ravel(), y.ravel()]).T
    Z = -gmm_model.score_samples(XX)
    Z = Z.reshape(x.shape)
    ax[1].contour(x, y, Z, cmap='coolwarm_r')

    # Add labels and title
    ax[0].set_xlabel('EOF 1')
    ax[0].set_ylabel('EOF 2')
    ax[1].set_xlabel('EOF 1')
    ax[1].set_ylabel('EOF 2')
    ax[0].set_title('Scatter plot of loadings')
    ax[1].set_title('Contour plot of fitted GMM')

    plt.tight_layout()
    plt.show()

    return ds


###############################################################################################
# function to plot the class labels on a map

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def plot_labels(ds,label_var_name='class_label'):
    sat_label = ds[label_var_name]

    # colors = ['blue', 'pink', 'green', 'orange', 'purple', 'brown', 'red', 'gray', 'olive', 'cyan']
    
    # create a figure and axis with Robinson projection
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(projection=ccrs.Robinson()), dpi=300)

    # add coastline and gridlines
    ax.add_feature(cfeature.COASTLINE)
    ax.gridlines()

    # create colormap with unique colors for each class label
    # cmap = mcolors.ListedColormap(colors[0:len(np.unique(sat_label))])
    unique_labels = np.unique(sat_label)
    cmap = plt.get_cmap('Accent', len(unique_labels))

    # plot heatmap with Robinson projection
    im = sat_label.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, shading='auto', add_colorbar=False)

    # set global extent for Robinson projection
    ax.set_global()
    
    # set title
    ax.set_title('Spatial distribution of class labels')

    # add colorbar
    bounds = np.arange(len(np.unique(sat_label))+1) - 0.5
    ticks = np.arange(len(np.unique(sat_label)))
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, boundaries=bounds, ticks=ticks)
    cbar.ax.set_yticklabels(np.unique(sat_label))
    cbar.ax.set_ylabel('Class Label')

    # show plot
    plt.show()

###############################################################################################
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def plot_weighted_average_curve(ds, label_var_name='class_label',dpi=100):
    unique_classes = np.unique(ds[label_var_name].values)
    nclasses = len(unique_classes)

    fig = plt.figure(figsize=(7, 4 * nclasses), constrained_layout=True, dpi=dpi)
    cmap = plt.get_cmap('Accent', len(unique_classes))

    spec = fig.add_gridspec(ncols=3, nrows=nclasses, width_ratios=[1, 2, 2])

    weighted_avg_curves = {}

    for i, class_label in enumerate(unique_classes):
        class_mask = ds[label_var_name] == class_label

        # Spatial Distribution Plot
        ax = fig.add_subplot(spec[i, 0], projection=ccrs.Robinson())
        ax.add_feature(cfeature.COASTLINE)
        ds[label_var_name].where(class_mask).plot(ax=ax, transform=ccrs.PlateCarree(),
                                                  cmap=cmap, vmin=0, vmax=nclasses,
                                                  add_colorbar=False)

        # Weighted Average SAT Curve
        ax = fig.add_subplot(spec[i, 1:])
        ds_subset = ds.where(class_mask, drop=True)
        
        weights_broadcasted = ds_subset['weight'].broadcast_like(ds_subset['sat'])
        sum_weighted_sat = (ds_subset['sat'] * weights_broadcasted).sum(dim=['lat', 'lon'])
        sum_weight_sat = weights_broadcasted.sum(dim=['lat', 'lon'])

        weighted_avg_sat = sum_weighted_sat / sum_weight_sat
        weighted_avg_curves[class_label] = weighted_avg_sat.data  # Store the weighted average curve
        ax.plot(ds['age'], weighted_avg_sat, color=cmap(class_label))
        ax.invert_xaxis()
        ax.set_title(f'Class {class_label}')

        # Add climate transitions timing
        HS1 = np.array([17480, 14692]) - 50  # convert to b1950
        BA = np.array([14692, 12896]) - 50
        YD = np.array([12896, 11703]) - 50

        for period, name in zip([HS1, BA, YD], ["HS1", "BA", "YD"]):
            ax.axvline(x=period[0], color='black', linestyle='--')
            ax.axvline(x=period[1], color='black', linestyle='--')
            ax.text(period.mean(), np.max(ax.get_ylim()), name, rotation=90, verticalalignment='top')

        ax.set_xlabel('Age (yr BP)')
        ax.set_ylabel('Weighted Average SAT (°C)')

    plt.show()
    return weighted_avg_curves


###########################################################################################
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import numpy as np

# def plot_weighted_average_curve(ds,dpi=100):
#     unique_classes = np.unique(ds['class_label'].values)
#     nclasses = len(unique_classes)

#     fig = plt.figure(figsize=(7, 4 * nclasses), constrained_layout=True, dpi=dpi)
#     # colors = ['blue', 'pink', 'green', 'orange', 'purple', 'brown', 'red', 'gray', 'olive', 'cyan']

#     cmap = plt.get_cmap('Accent', len(unique_classes))

#     spec = fig.add_gridspec(ncols=3, nrows=nclasses, width_ratios=[1, 2, 2])

#     for i, class_label in enumerate(unique_classes):
#         class_mask = ds['class_label'] == class_label
#         # cmap = mcolors.ListedColormap(['#ffffff', cmap[class_label]])

#         # Spatial Distribution Plot
#         ax = fig.add_subplot(spec[i, 0], projection=ccrs.Robinson())
#         ax.add_feature(cfeature.COASTLINE)
#         # ax.add_feature(cfeature.BORDERS, linestyle=':')
#         # ds['class_label'].where(class_mask).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap(class_label), add_colorbar=False)
#         ds['class_label'].where(class_mask).plot(ax=ax, transform=ccrs.PlateCarree(),
#                                             cmap=cmap, vmin=0, vmax=nclasses,
#                                             add_colorbar=False)
#         # ax.set_title(f'Class {class_label} Distribution')

#         # Weighted Average SAT Curve
#         ax = fig.add_subplot(spec[i, 1:])
#         ds_subset = ds.where(class_mask, drop=True)
        
#         weights_broadcasted = ds_subset['weight'].broadcast_like(ds_subset['sat'])
#         sum_weighted_sat = (ds_subset['sat'] * weights_broadcasted).sum(dim=['lat', 'lon'])
#         sum_weight_sat = weights_broadcasted.sum(dim=['lat', 'lon'])

#         weighted_avg_sat = sum_weighted_sat / sum_weight_sat
#         # ax.plot(ds['age'], weighted_avg_sat, color=cmap[class_label])
#         ax.plot(ds['age'], weighted_avg_sat, color=cmap(class_label))
#         # invert the x-axis
#         ax.invert_xaxis()
    
#         ax.set_title(f'Class {class_label} Weighted Average SAT Curve')

#         # Add climate transitions timing
#         HS1 = np.array([17480, 14692]) - 50  # convert to b1950
#         BA = np.array([14692, 12896]) - 50
#         YD = np.array([12896, 11703]) - 50

#         for period, name in zip([HS1, BA, YD], ["HS1", "BA", "YD"]):
#             ax.axvline(x=period[0], color='black', linestyle='--')
#             ax.axvline(x=period[1], color='black', linestyle='--')
#             ax.text(period.mean(), np.max(ax.get_ylim()), name, rotation=90, verticalalignment='top')

#         ax.set_xlabel('Age (yr BP)')
#         ax.set_ylabel('SAT (°C)')

#     plt.show()

###########################################################################################
