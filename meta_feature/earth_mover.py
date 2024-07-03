from scipy.stats import wasserstein_distance
import numpy as np

def compute_histograms(dataset, bins=256, range=(0, 1)):
    histograms = []
    for data in dataset:
        img, _ = data
        img = img.numpy().flatten()  # Convert to numpy array and flatten
        hist, _ = np.histogram(img, bins=bins, range=range, density=True)
        histograms.append(hist)
    return np.array(histograms)

def calculate_emd(histograms1, histograms2):
    avg_hist1 = np.mean(histograms1, axis=0)
    avg_hist2 = np.mean(histograms2, axis=0)
    emd = wasserstein_distance(avg_hist1, avg_hist2)
    return emd

