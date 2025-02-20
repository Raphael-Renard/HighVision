import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from clustering import getPredictionFromThreshold

def p_r_class(predicted_y, y):
    y = np.array(y)
    predicted_y = np.array(predicted_y)

    l_y = int(max(y)) + 1
    l_predicted_y = int(max(predicted_y)) + 1

    class_precisions = []
    class_recalls = []

    for real in range(l_y):
        precisions = []
        recalls = []

        for predicted in range(l_predicted_y):
            true = np.argwhere(y == real).flatten()
            positive = np.argwhere(predicted_y == predicted).flatten()
            TP = len(np.intersect1d(true, positive))
            recall = TP / len(true)
            if recall > 0:
                precisions.append(TP / len(positive))
                recalls.append(recall)
        
        # print(sum(recalls)) # should be 1
        precision = np.mean(np.multiply(precisions, recalls))
        recall = np.mean(np.square(recalls))

        class_precisions.append(precision)
        class_recalls.append(recall)
    
    return class_precisions, class_recalls

def p_r_f1(predicted_y, y):
    TP = 0
    FP = 0
    FN = 0
    # TN = 0

    for i in range(len(y)):
        for j in range(i+1, len(y)):
            if y[i] == y[j]:
                if predicted_y[i] == predicted_y[j]:
                    TP += 1
                else:
                    FN += 1
            else:
                if predicted_y[i] == predicted_y[j]:
                    FP += 1
                # else: # not used
                #    TN += 1

    if TP == 0 :
        precision = 0
        recall = 0
        f1 = 0
    else :
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def p_r_f1_byThresholds(thresholds, distance, y):
    precisions = []
    recalls = []
    f1s = []
    for threshold in tqdm(thresholds, desc="Thresholds"):
        predicted_y = getPredictionFromThreshold(threshold, distance)
        precision, recall, f1 = p_r_f1(predicted_y, y)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return precisions, recalls, f1s

def p_r_class_byThresholds(thresholds, distance, y):
    precisions_per_class = []
    recalls_per_class = []
    for threshold in tqdm(thresholds, desc="Thresholds"):
        predicted_y  = getPredictionFromThreshold(threshold, distance)
        precisions, recalls = p_r_class(predicted_y, y)
        precisions_per_class.append(precisions)
        recalls_per_class.append(recalls)
    return precisions_per_class, recalls_per_class

def fei_mAP(y, distance):
    y = np.array(y)
    similar_images = np.argsort(distance)[:,1:] # remove self
    APs = []

    for i in range(len(y)):
        true_similars = np.argwhere(y == y[i]).flatten()
        l = len(true_similars) - 1
        if l == 0:
            continue
        predicted_similar = similar_images[i][:l]
        true_similar = 0
        precision_sum = 0
        for j in range(len(predicted_similar)):
            if predicted_similar[j] in true_similars:
                true_similar += 1
                precision_sum += true_similar / (j + 1)
        APs.append(precision_sum / l)
    return np.mean(APs)

def goncalves_mAP(precisions_per_class, recalls_per_class):
    APs_per_class = []
    precisions_per_class = np.array(precisions_per_class)
    recalls_per_class = np.array(recalls_per_class)

    for i in range(precisions_per_class.shape[1]):
        APs_per_class.append(np.trapezoid(precisions_per_class[:,i], x=recalls_per_class[:,i]))
    return np.mean(np.array(APs_per_class))

def pr_curve(precisions, recalls, f1s = None, other = None, save=None):
    AP = np.trapezoid(precisions, x=recalls)
    plt.plot(recalls, precisions)
    plt.xlim(0,1)
    plt.ylim(0,1)
    if f1s is not None:
        best_threshold = np.argmax(f1s)
        plt.plot(recalls[best_threshold], precisions[best_threshold], 'ro', label='best threshold by F1 (' + "{:.3f}".format(np.max(f1s)) + ')')
        plt.legend()
    if other is None:
        plt.title('Precision Recall Curve\nAP = ' + "{:.3f}".format(AP))
    else:
        plt.title('Precision Recall Curve\nAP = ' + "{:.3f}".format(AP) + '\n' + other[0] + ' = ' + "{:.3f}".format(other[1]))
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    if save:
        plt.savefig(save)
    else:
        plt.show()

    if f1s is not None:
        return AP,best_threshold
    return AP,None