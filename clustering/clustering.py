from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt

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

def visualiseClustering(path_npy, images, max_size=5):
    labels = np.load(path_npy)
    groups = []
    for _ in range(int(max(labels)) + 1):
        groups.append([])

    for i in range(len(labels)):
        groups[int(labels[i])].append(i)
    
    for i in range(len(groups)):
        print("Group",i,":", groups[i])
        if len(groups[i]) > 1 and len(groups[i]) <= max_size:
            fig, axes = plt.subplots(1, len(groups[i]), figsize=(20, 20))
            for ax, im in enumerate(groups[i]):
                axes[ax].imshow(images[im], cmap='gray')
                axes[ax].axis('off')
            plt.show()