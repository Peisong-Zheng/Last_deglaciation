import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def plot_ens_labels(labels_Ens, figsize=(10, 6), dpi=300, panel_label='a'):
    """
    Plots a heatmap of ensemble labels.

    :param labels_Ens: List of dictionaries containing label arrays
    :param figsize: Tuple, the size of the figure (width, height)
    :param cmap_name: String, the name of the matplotlib colormap to use
    """
    # Extract the label arrays and stack them into a 2D numpy array
    label_arrays = [entry['labels'] for entry in labels_Ens]
    label_matrix = np.vstack(label_arrays)

    # Create the heatmap
    fig=plt.figure(figsize=figsize,dpi=dpi)  # Adjust the figure size as needed
    unique_labels = np.unique(label_matrix)
    
    if len(unique_labels) <= 4:
        custom_colors = [
        (0.4980392156862745, 0.788235294117647, 0.4980392156862745),
        (0.9921568627450981, 0.7529411764705882, 0.5254901960784314),
        (0.9411764705882353, 0.00784313725490196, 0.4980392156862745),
        (0.27450980392156865, 0.5098039215686274, 0.7058823529411765),]

        # Create a ListedColormap object with your custom colors
        cmap = ListedColormap(custom_colors)   
    else:
        cmap = plt.get_cmap('Accent', len(unique_labels))
    
    # add panel label
    plt.text(-0.07, 1.05, panel_label, transform=plt.gca().transAxes, size=16, weight='bold')

    # cmap = plt.get_cmap(cmap_name, len(unique_labels))

    # Set colorbar ticks
    boundaries = np.arange(len(unique_labels) + 1) - 0.5
    ticks = np.arange(len(unique_labels))

    # Create the heatmap with adjusted colorbar
    ax = sns.heatmap(label_matrix, cmap=cmap, cbar_kws={'label': 'Class Label', 'ticks': ticks, 'boundaries': boundaries, 'spacing': 'uniform'})

    # Adjust colorbar tick labels
    cbar = ax.collections[0].colorbar
    # cbar.set_ticks(ticks + 0.5)  # Set to middle of each segment
    cbar.set_ticklabels(ticks + 1)  # Increment labels by 1

    plt.xlabel('Grid Index')
    plt.ylabel('Ens Index')

    plt.show()
    return fig

# ##########################################################################################
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.colors import ListedColormap

# def plot_ens_labels(labels_Ens, figsize=(10, 6), dpi=300, colors=['blue', 'pink', 'green', 'orange', 'purple', 'brown', 'red', 'gray', 'olive', 'cyan']):
#     """
#     Plots a heatmap of ensemble labels.

#     :param labels_Ens: List of dictionaries containing label arrays
#     :param figsize: Tuple, the size of the figure (width, height)
#     :param dpi: Integer, the dots per inch setting for the figure
#     :param colors: List of strings, specifying the colors to use for the color map
#     """
#     # Extract the label arrays and stack them into a 2D numpy array
#     label_arrays = [entry['labels'] for entry in labels_Ens]
#     label_matrix = np.vstack(label_arrays)

#     # Create the heatmap
#     plt.figure(figsize=figsize,dpi=dpi)  # Adjust the figure size as needed
#     unique_labels = np.unique(label_matrix)
    
#     if colors is not None:
#         if len(colors) < len(unique_labels):
#             raise ValueError("The number of colors provided is less than the number of unique labels.")
#         cmap = ListedColormap(colors[:len(unique_labels)])
#     else:
#         cmap = plt.get_cmap('Accent', len(unique_labels))

#     # Set colorbar ticks
#     boundaries = np.arange(len(unique_labels) + 1) - 0.5
#     ticks = np.arange(len(unique_labels))

#     # Create the heatmap with adjusted colorbar
#     ax = sns.heatmap(label_matrix, cmap=cmap, cbar_kws={'label': 'Class Label', 'ticks': ticks, 'boundaries': boundaries, 'spacing': 'uniform'})
#     plt.xlabel('Grid Index')
#     plt.ylabel('Ens Index')

#     plt.show()



#################################################################
import numpy as np

def align_labels4two_iters(reference_labels, labels_to_align):
    unique_ref_labels = np.unique(reference_labels)
    unique_align_labels = np.unique(labels_to_align)
    
    n_classes_ref = len(unique_ref_labels)
    n_classes_align = len(unique_align_labels)

    # Ensure the number of unique classes is the same in both label sets
    if n_classes_ref != n_classes_align:
        raise ValueError("The number of unique classes in the reference and labels to align must be the same.")

    n_classes = n_classes_ref

    # Create a mapping based on overlap
    mapping = {}
    for ref_label in unique_ref_labels:
        overlaps = [] # for each ref_label, calculate the sum of overlap with each align_label
        for align_label in unique_align_labels:
            overlap = np.sum((labels_to_align == align_label) & (reference_labels == ref_label))
            overlaps.append(overlap)

        # Get the align label with the max overlap for the current reference label
        max_overlap_label = unique_align_labels[np.argmax(overlaps)]
        
        if max_overlap_label in mapping.values():
            # Handle the situation where a label has already been mapped due to identical overlaps
            # For example, for two ref_labels, they corresponds to the same align_label
            available_labels = list(set(unique_align_labels) - set(mapping.values())) # find the labels in unique_align_labels that have not been mapped
            overlaps_available = [overlaps[i] for i in range(n_classes) if unique_align_labels[i] in available_labels]
            max_overlap_label = available_labels[np.argmax(overlaps_available)]

        mapping[ref_label] = max_overlap_label

    # Apply the mapping to the labels_to_align array
    aligned_labels = np.copy(labels_to_align)
    for ref, align in mapping.items():
        aligned_labels[labels_to_align == align] = ref

    return aligned_labels


def align_labels(results,reference_labels):
    n_iterations=len(results)

    # # read results_sklearn from a pickle file
    # import pickle
    # with open('data_fig1/results_sklearn.pkl', 'rb') as f:
    #     results_ref = pickle.load(f)

    # reference_labels = results[0]['labels']
    for i in range(0, n_iterations):
        aligned_labels = align_labels4two_iters(reference_labels, results[i]['labels'])
        results[i]['labels'] = aligned_labels
    return results

#################################################################
import numpy as np
def measure_consistency(labels,threshold_of_consistency=0.05):
    """
    Measure the consistency of label assignments compared to the first iteration.

    Parameters:
    - labels: List of dictionaries, each containing a 'labels' key with numpy array of labels.

    Returns:
    - consistency_ratio: Proportion of iterations where labels are consistent with the first iteration.
    """

    reference_labels = labels[0]['labels']
    n_samples = len(reference_labels)
    print('n_samples=',n_samples)
    threshold = threshold_of_consistency * n_samples  # 5% of the total samples
    print('threshold=',threshold)
    
    consistent_iterations = 0

    for result in labels:
        # Compute the number of differing labels compared to the reference
        diff_labels_count = np.sum(result['labels'] != reference_labels)
        print('diff_labels_count',diff_labels_count)
        
        if diff_labels_count <= threshold:
            consistent_iterations += 1

    consistency_ratio = consistent_iterations / len(labels)
    
    return consistency_ratio


#################################################################