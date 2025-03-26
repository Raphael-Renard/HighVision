from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import umap

def getPredictionFromThreshold(threshold, distanceMat):
    matchesModel = AgglomerativeClustering(distance_threshold=threshold, metric='precomputed', n_clusters=None, linkage='average').fit(distanceMat)
    predicted_labels = matchesModel.labels_
    predicted_groups = []
    for i in range(np.max(predicted_labels)+1):
        predicted_groups.append([])
    for i in range(len(predicted_labels)):
        predicted_groups[predicted_labels[i]].append(i)
    predicted_groups = sorted(predicted_groups, key=lambda x: np.min(x))
    predicted_y = np.empty(distanceMat.shape[0])
    for i in range(len(predicted_groups)):
        for j in predicted_groups[i]:
            predicted_y[j] = i
    return predicted_y

def visualiseClustering(labels, images, max_size=5):
    groups = []
    for _ in range(int(max(labels)) + 1):
        groups.append([])

    for i in range(len(labels)):
        groups[int(labels[i])].append(i)
    
    for i in range(len(groups)):
        if len(groups[i]) >= 1:
            print("Group",i,":", groups[i])
            if len(groups[i]) > 1 and len(groups[i]) <= max_size:
                fig, axes = plt.subplots(1, len(groups[i]), figsize=(20, 20))
                for ax, im in enumerate(groups[i]):
                    axes[ax].imshow(images[im], cmap='gray')
                    axes[ax].axis('off')
                plt.show()

def visualiseClusteringRepresentation(labels, representation, rect=None):
    if representation.shape[1] != 2:
        representation = umap.UMAP(n_components=2, random_state=42).fit_transform(representation)

    scatter_x = representation[:,0]
    scatter_y = representation[:,1]

    unique_labels = np.unique(labels)
    colors = cm.get_cmap('tab20b', len(unique_labels)).colors 
    label_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    for label in unique_labels:
        ix = np.where(labels == label)
        plt.scatter(scatter_x[ix], scatter_y[ix], label=label, color=label_color_map[label])
    
    points = None
    if rect is not None:
        (x_min, y_min), (x_max, y_max) = rect
        l_x = x_max - x_min
        l_y = y_max - y_min
        points = np.where((scatter_x >= x_min) & (scatter_x <= x_max) & (scatter_y >= y_min) & (scatter_y <= y_max))[0]
        plt.gca().add_patch(plt.Rectangle((x_min, y_min), l_x, l_y, ls="--", lw=2, ec="r", fc="none"))
    
    plt.show()
    return points