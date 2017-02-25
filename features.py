import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import glob
import pickle
import os


class DataLoader:
    def __init__(self):
        self.pickle_filename = 'data.p'

    def load_data(self, path, use_pickle=True, reset_pickle=False):
        """
        Loads training data from specified directories.  All images are stored using RGB color space.
        """
        if use_pickle and os.path.exists(self.pickle_filename) and not reset_pickle:
            image_data = self.load_from_pickle(path)
            # if path was cached and we found features, return them
            if image_data is not None:
                return image_data

        image_data = self.load_from_filesystem(path)

        if use_pickle or reset_pickle:
            self.save_to_pickle(image_data, path)

        return image_data

    def save_to_pickle(self, image_data, path):
        """
        Saves data to specified pickle file.
        """
        data = {}
        try:
            with open(self.pickle_filename, 'rb') as file:
                data = pickle.load(file)
        except FileNotFoundError:
            print("Creating new pickle file")

        with open(self.pickle_filename, 'wb') as file:
            data[path] = image_data
            pickle.dump(data, file)

    def load_from_pickle(self, path):
        """
        Loads saved data from specified pickle file.
        """
        with open(self.pickle_filename, 'rb') as file:
            data = pickle.load(file)
            if path not in data:
                return None

            return data[path]

    def load_from_filesystem(self, path):
        image_data = []

        for filename in glob.iglob(path, recursive=True):
            img = cv2.imread(filename)
            image_data.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return image_data


from skimage.feature import hog

class FeatureExtractor:

    def extract_features(self, feature_images, colorspace='RGB', spatial_size=(32, 32), hist_bins=32,
                         hist_range=(0, 256), orientations=9, hog_pixels_per_cell=8, hog_cells_per_block=2,
                         hog_visualize=False, feature_vector_hog=True):
        features = []
        for src_img in feature_images:
            feature_img = self.convert_colorspace(src_img, colorspace)
            gray_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY) * 1.0

            spatial_features = self.bin_spatial(feature_img, size=spatial_size)
            hist_features = self.color_hist(feature_img, nbins=hist_bins, bins_range=hist_range)

            hog_features = self.hog_features(gray_img, orientations=orientations,
                                             pixels_per_cell=hog_pixels_per_cell, cells_per_block=hog_cells_per_block,
                                             visualize=hog_visualize, feature_vector=feature_vector_hog)
            feature = np.concatenate((spatial_features, hist_features, hog_features))
            features.append(feature)

        return features

    @staticmethod
    def bin_spatial(img, size=(32, 32)):
        # Resize and use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    @staticmethod
    def color_hist(img, nbins=32, bins_range=(0.256)):
        # Compute the histogram of the image channels separately
        ch1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        ch2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        ch3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

        # Generating bin centers
        bin_edges = ch1_hist[1]
        bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2

        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))

        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    @staticmethod
    def hog_features(img, orientations=9, pixels_per_cell=8, cells_per_block=2, visualize=False, feature_vector=True):
        if visualize == True:
            features, hog_image = hog(img, orientations=orientations, pixels_per_cell=(pixels_per_cell,pixels_per_cell),
                                      cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=False,
                                      visualise=True, feature_vector=False)
            return features, hog_image
        else:
            features = hog(img, orientations=orientations, pixels_per_cell=(pixels_per_cell,pixels_per_cell),
                           cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=False,
                           visualise=False, feature_vector=feature_vector)
            return features

    @staticmethod
    def convert_colorspace(img, color_space):
        if color_space == 'RGB':
            return np.copy(img)
        elif color_space == 'HSV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            print("Unknown or unsupported color space specified, defaulting to RGB.")
            return np.copy(img)


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC


class SupportVectorClassifier:
    # TODO, may need to worry about time series of images taken from video, since training data and test data could be very similar

    def __init__(self, vehicle_features, non_vehicle_features):
        vehicle_labels = np.ones(len(vehicle_features))
        non_vehicle_labels = np.zeros(len(non_vehicle_features))

        # build labels
        y = np.hstack((vehicle_labels, non_vehicle_labels))

        # stack features and scale them
        X = np.vstack((vehicle_features, non_vehicle_features))
        X = X.astype(np.float64)

        X_scaler = StandardScaler().fit(X)
        X = X_scaler.transform(X)

        # randomize features/labels
        self.X, self.y = shuffle(X, y)

        # setup SVC
        self.svc = LinearSVC()

    def train_and_score(self, test_size=0.25):
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=rand_state)

        # train the linear support vector classifier
        self.svc.fit(X_train, y_train)

        return self.svc.score(X_test, y_test)
