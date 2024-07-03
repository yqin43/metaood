import numpy as np
import pickle

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

import scipy.stats as stats
from scipy.stats import moment
from scipy.stats import normaltest
from scipy.stats import skew, kurtosis
import time

def sobel_filter(image):
    # Define Sobel operator kernels
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    # Convert image to grayscale
    image_gray = image.mean(dim=0, keepdim=True)

    # Apply Sobel operator
    edge_x = F.conv2d(image_gray.unsqueeze(0), kernel_x, padding=1)
    edge_y = F.conv2d(image_gray.unsqueeze(0), kernel_y, padding=1)

    # Combine edges
    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2).squeeze(0)

    return edge

def compute_hu_moments(edge_image):
    # Compute Moments
    moments = cv2.moments(edge_image)
    # Compute Hu Moments
    huMoments = cv2.HuMoments(moments)
    # Log scale transform
    huMoments = -1 * np.sign(huMoments) * np.log10(np.abs(huMoments))
    return huMoments

def apply_edge_detection_and_compute_hu(image_tensor):
    # Convert PyTorch tensor to numpy array
    image_np = image_tensor.numpy().transpose((1, 2, 0)) * 255
    image_np = image_np.astype(np.uint8)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Sobel filter
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.hypot(sobelx, sobely)
    sobel_norm = np.uint8(255 * sobel / np.max(sobel))
    
    # Compute Hu Moments for both edge detections
    huMoments_sobel = compute_hu_moments(sobel_norm)
    
    return huMoments_sobel

def count_white_pixels(edge_image, threshold=0.5):
    # Count pixels above the threshold
    return torch.sum(edge_image > threshold).item()

def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def rgb_to_hsv(image):
    assert image.ndim == 3
    pil_img = transforms.ToPILImage()(image)
    hsv_img = pil_img.convert('HSV')
    return transforms.ToTensor()(hsv_img)

def calculate_spearman_correlation(images):
    images = images.permute(0, 2, 3, 1)  # Change to NHWC format
    flat_images = images.reshape(-1, 3).numpy()  # Flatten and convert to NumPy for correlation calculation
    corr_matrix = stats.spearmanr(flat_images, axis=0)[0]  # Spearman correlation matrix
    return corr_matrix

def calculate_entropy(img_tensor):
    # Convert to numpy array and flatten
    img = img_tensor.numpy().flatten()
    # Compute histogram
    hist, _ = np.histogram(img, bins=256, range=(0, 1))
    # Normalize to create a probability distribution
    p = hist / hist.sum()
    # Calculate entropy
    entropy = -np.sum([p_i*np.log2(p_i) for p_i in p if p_i > 0])
    return entropy

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = np.add(array, 0.0000001, casting="unsafe")
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def flatten_diagonally(x, diags=None):
    diags = np.array(diags)
    if x.shape[1] > x.shape[0]:
        diags += x.shape[1] - x.shape[0]
    n = max(x.shape)
    ndiags = 2 * n - 1
    i, j = np.indices(x.shape)
    d = np.array([])
    for ndi in range(ndiags):
        if diags != None:
            if not ndi in diags:
                continue
        d = np.concatenate((d, x[i == j + (n - 1) - ndi]))
    return d

def list_process(x, r_min=True, r_max=True, r_mean=True, r_std=True,
                 r_skew=True, r_kurtosis=True):
    """Return statistics of a list

    Parameters
    ----------
    x
    r_min
    r_max
    r_mean
    r_std
    r_skew
    r_kurtosis

    Returns
    -------

    """
    x = np.asarray(x).reshape(-1, 1)
    return_list = []

    if r_min:
        return_list.append(np.nanmin(x))

    if r_max:
        return_list.append(np.nanmax(x))

    if r_mean:
        return_list.append(np.nanmean(x))

    if r_std:
        return_list.append(np.nanstd(x))

    if r_skew:
        return_list.append(skew(x, nan_policy='omit')[0])

    if r_kurtosis:
        return_list.append(kurtosis(x, nan_policy='omit')[0])

    return return_list


def list_process_name(var):
    return [var + '_min', var + '_max', var + '_mean', var + '_std',
            var + '_skewness', var + '_kurtosis']

################################# Statistical Feature #######################################
def extract_meta_features(loader):
    meta_vec = []
    meta_vec_name = []
    start_t = time.time()
    for data in loader:
        images, labels = data
        X = images.numpy()
        y = labels.numpy()
        print('data fininshed loading')
        sample_mean = np.mean(X)
        sample_median = np.median(X)
        sample_var = np.var(X)
        sample_min = np.min(X)
        sample_max = np.max(X)
        sample_std = np.std(X)

        q1, q25, q75, q99 = np.percentile(X, [0.01, 0.25, 0.75, 0.99])
        iqr = q75 - q25

        normalized_mean = sample_mean / sample_max
        normalized_median = sample_median / sample_max
        sample_range = sample_max - sample_min
        sample_gini = gini(X)
        med_abs_dev = np.median(np.absolute(X - sample_median))
        avg_abs_dev = np.mean(np.absolute(X - sample_mean))
        quant_coeff_disp = (q75 - q25) / (q75 + q25)
        coeff_var = sample_var / sample_mean

        n_samples = len(y)
        dim = X.shape[1]
        n_features = np.size(X[0])
        num_classes = len(np.unique(y))

        meta_vec.extend(
            [sample_mean, sample_median, sample_var, sample_min, sample_max,
            sample_std,
            q1, q25, q75, q99, iqr, normalized_mean, normalized_median,
            sample_range, sample_gini,
            med_abs_dev, avg_abs_dev, quant_coeff_disp, coeff_var,
            n_samples,dim, n_features, num_classes
            ])
        meta_vec_name.extend(
            ['sample_mean', 'sample_median', 'sample_var', 'sample_min',
            'sample_max', 'sample_std',
            'q1', 'q25', 'q75', 'q99', 'iqr', 'normalized_mean',
            'normalized_median', 'sample_range', 'sample_gini',
            'med_abs_dev', 'avg_abs_dev', 'quant_coeff_disp', 'coeff_var',
            'num_samples','dim', 'num_features', 'num_classes',
            ])
        
        normality_k2, normality_p = normaltest(X)
        is_normal_5 = (normality_p < 0.05).astype(int)
        is_normal_1 = (normality_p < 0.01).astype(int)

        meta_vec.extend(list_process(normality_p))
        meta_vec.extend(list_process(is_normal_5))
        meta_vec.extend(list_process(is_normal_1))

        meta_vec_name.extend(list_process_name('normality_p'))
        meta_vec_name.extend(list_process_name('is_normal_5'))
        meta_vec_name.extend(list_process_name('is_normal_1'))

        moment_5 = moment(X, moment=5)
        moment_6 = moment(X, moment=6)
        moment_7 = moment(X, moment=7)
        moment_8 = moment(X, moment=8)
        moment_9 = moment(X, moment=9)
        moment_10 = moment(X, moment=10)
        meta_vec.extend(list_process(moment_5))
        meta_vec.extend(list_process(moment_6))
        meta_vec.extend(list_process(moment_7))
        meta_vec.extend(list_process(moment_8))
        meta_vec.extend(list_process(moment_9))
        meta_vec.extend(list_process(moment_10))
        meta_vec_name.extend(list_process_name('moment_5'))
        meta_vec_name.extend(list_process_name('moment_6'))
        meta_vec_name.extend(list_process_name('moment_7'))
        meta_vec_name.extend(list_process_name('moment_8'))
        meta_vec_name.extend(list_process_name('moment_9'))
        meta_vec_name.extend(list_process_name('moment_10'))

        print(time.time()-start_t)
        # note: this is for each dimension == the number of dimensions
        skewness_list = skew(X).reshape(-1, 1)
        skew_values = list_process(skewness_list)
        meta_vec.extend(skew_values)
        meta_vec_name.extend(list_process_name('skewness'))

        # note: this is for each dimension == the number of dimensions
        kurtosis_list = kurtosis(X)
        kurtosis_values = list_process(kurtosis_list)
        meta_vec.extend(kurtosis_values)
        meta_vec_name.extend(list_process_name('kurtosis'))
        print(time.time()-start_t)
        # Convert RGB images to HSV
        X_torch = torch.from_numpy(X)
        hsv_images = torch.stack([rgb_to_hsv(img) for img in X_torch])

        # Calculate intensity (average of RGB channels)
        intensity_images = X_torch.mean(dim=1, keepdim=True)

        # Flatten and concatenate all channels
        all_channels = torch.cat([X_torch, hsv_images, intensity_images], dim=1)
        all_channels_flat = all_channels.reshape(-1, all_channels.shape[1]).numpy()

        # Calculate Spearman correlation
        corr_matrix = stats.spearmanr(all_channels_flat, axis=0)[0]
        meta_vec.extend(corr_matrix.flatten().tolist())
        meta_vec_name.extend(list_process_name('corr'))
        print(time.time()-start_t)
        # mean of HSV
        hsv_mean = torch.mean(hsv_images)
        # std
        hsv_std = torch.std(hsv_images)
        intensity_std = torch.std(intensity_images)

        meta_vec.extend([hsv_mean.item(), hsv_std.item(), intensity_std.item()])
        meta_vec_name.extend(['hsv_mean', 'hsv_std', 'intensity_std', 'intensity_entropy'])
        print(time.time()-start_t)
        entropies = []
        for image in images:
            entropy = calculate_entropy(image)
            entropies.append(entropy)
        meta_vec.append(sum(entropies)/len(entropies))
        meta_vec_name.append('entropy')
        
        # std of histgram
        hist, _ = np.histogram(X, bins=256, range=(0, 1))
        hist_std = np.std(hist)
        hist_hsv = torch.histogram(hsv_images)
        hist_intensity = torch.histogram(intensity_images)
        hist_hsv_std = torch.std(hist_hsv.hist)
        hist_intensity_std = torch.std(hist_intensity.hist)
        meta_vec.extend([hist_std, hist_hsv_std.item(), hist_intensity_std.item()])
        meta_vec_name.extend(['hist_std', 'hist_hsv_std', 'hist_intensity_std'])
        print(time.time()-start_t)
        white_pixel_counts = []
        humoments_sobel = []
        for image in X_torch:
            sobel_image = sobel_filter(image)
            white_count = count_white_pixels(sobel_image)
            white_pixel_counts.append(white_count)

            huMoments_sobel = apply_edge_detection_and_compute_hu(image)
            humoments_sobel.append(np.concatenate(huMoments_sobel).tolist())

        print(time.time()-start_t)
        # Calculate average number of white pixels
        average_white_pixels = sum(white_pixel_counts) / len(white_pixel_counts)
        meta_vec.append(average_white_pixels)
        average_humoments_sobel = np.mean(humoments_sobel, axis=0)
        meta_vec.extend(average_humoments_sobel)
        meta_vec_name.extend(['average_white_pixels', 'average_humoments_sobel'])
        
        # Co-ocurrence matrix
        # Function to calculate GLCM and its properties
        cooc_contrast = []
        cooc_dissimilarity = []
        cooc_homogeneity = []
        cooc_energy = []
        cooc_correlation = []
        cooc_entropy = []
        def calculate_glcm_properties(image, distances, angles, levels=256):
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = img_as_ubyte(image)  # Convert image to 8-bit unsigned integers
            glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
            cooc_contrast.append(graycoprops(glcm, 'contrast')[0, 0])
            cooc_dissimilarity.append(graycoprops(glcm, 'dissimilarity')[0, 0])
            cooc_homogeneity.append(graycoprops(glcm, 'homogeneity')[0, 0])
            cooc_energy.append(graycoprops(glcm, 'energy')[0, 0])
            cooc_correlation.append(graycoprops(glcm, 'correlation')[0, 0])
            cooc_entropy.append(-np.sum(glcm * np.log2(glcm + 1e-10)))  # Adding small constant to avoid log(0)  
        
        fft_entropy = []
        fft_inertia = []
        fft_energy = []
        fft_homogeneity = []

        def compute_fft_properties(image):
            # Compute FFT
            fft = torch.fft.fft2(image)
            # Compute magnitude spectrum and shift the zero frequency component to the center
            magnitude = torch.fft.fftshift(torch.abs(fft))
            
            # Normalize and scale to 0-255
            magnitude = magnitude - magnitude.min()
            magnitude = magnitude / magnitude.max() * 255.0
            magnitude = magnitude.type(torch.uint8)
            
            # Convert to numpy array for GLCM calculation
            magnitude_np = magnitude.numpy().squeeze()
            
            # Calculate GLCM and return texture properties
            glcm = graycomatrix(magnitude_np, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            fft_entropy.append(-np.sum(glcm * np.log2(glcm + 1e-10)))  # Entropy
            fft_inertia.append(graycoprops(glcm, 'contrast')[0, 0])     # Inertia (Contrast)
            fft_energy.append(graycoprops(glcm, 'energy')[0, 0])       # Energy
            fft_homogeneity.append(graycoprops(glcm, 'homogeneity')[0, 0])  # Homogeneity
        
        def rgb2gray(rgb):
            weights = np.array([0.2989, 0.5870, 0.1140])
            image_gray = np.tensordot(weights, rgb, axes=([0], [0]))
            image_gray = image_gray[np.newaxis, :, :]
            return image_gray

        for img in X:
            # Convert RGB to grayscale
            img = rgb2gray(img)
            image_np = img.squeeze()
            # Calculate GLCM properties
            calculate_glcm_properties(image_np, distances=[1], angles=[0])
            compute_fft_properties(torch.Tensor(img))
        print(time.time()-start_t)
        # coor
        meta_vec.extend([np.mean(cooc_contrast), np.mean(cooc_dissimilarity), np.mean(cooc_homogeneity),
                         np.mean(cooc_energy), np.mean(cooc_correlation), np.mean(cooc_entropy),
                         np.std(cooc_contrast), np.std(cooc_dissimilarity), np.std(cooc_homogeneity),
                         np.std(cooc_energy), np.std(cooc_correlation), np.std(cooc_entropy),])
        meta_vec_name.extend(['cooc_contrast_mean', 'cooc_dissimilarity_mean', 'cooc_homogeneity_mean',
                              'cooc_energy_mean', 'cooc_correlation_mean', 'cooc_entropy_mean',
                              'cooc_contrast_std', 'cooc_dissimilarity_std', 'cooc_homogeneity_std',
                              'cooc_energy_std', 'cooc_correlation_std', 'cooc_entropy_std'])
        # fft
        meta_vec.extend([np.mean(fft_entropy), np.mean(fft_inertia), np.mean(fft_energy), np.mean(fft_homogeneity),
                         np.std(fft_entropy), np.std(fft_inertia), np.std(fft_energy), np.std(fft_homogeneity)])
        meta_vec_name.extend(['fft_entropy_mean', 'fft_inertia_mean', 'fft_energy_mean', 'fft_homogeneity_mean',
                              'fft_entropy_std', 'fft_inertia_std', 'fft_energy_std', 'fft_homogeneity_std'])

        return meta_vec, meta_vec_name