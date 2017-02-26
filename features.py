from threading import Thread

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from copy import deepcopy
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
import time


class FeatureExtractor(Thread):
    def __init__(self, colorspace='RGB', spatial_size=(32, 32), hist_bins=32,
                 orientations=9, hog_pixels_per_cell=8, hog_cells_per_block=2,
                 feature_vector_hog=True, use_hog=True, use_bin_spatial=True, use_color_hist=True):
        Thread.__init__(self)
        self.features = []

        self.colorspace = colorspace
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orientations = orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        self.feature_vector_hog = feature_vector_hog
        self.use_hog = use_hog
        self.use_bin_spatial = use_bin_spatial
        self.use_color_hist = use_color_hist

    def get_features(self, feature_images):
        self.features = []
        for src_img in feature_images:
            feature_img = self.convert_colorspace(src_img, self.colorspace)

            if self.use_bin_spatial:
                spatial_features = self.bin_spatial(feature_img, size=self.spatial_size)
            else:
                spatial_features = []

            if self.use_color_hist:
                hist_features = self.color_hist(feature_img, nbins=self.hist_bins)
            else:
                hist_features = []

            if self.use_hog:
                hog_features = self.hog_features(feature_img, orientations=self.orientations,
                                                 pixels_per_cell=self.hog_pixels_per_cell,
                                                 cells_per_block=self.hog_cells_per_block,
                                                 feature_vector=self.feature_vector_hog)
            else:
                hog_features = []

            feature = np.concatenate((spatial_features, hist_features, hog_features))
            self.features.append(feature)

        return self.features


    @staticmethod
    def bin_spatial(img, size=(32, 32)):
        # Resize and use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    @staticmethod
    def color_hist(img, nbins=32):
        # Compute the histogram of the image channels separately
        ch1_hist = np.histogram(img[:, :, 0], bins=nbins)
        ch2_hist = np.histogram(img[:, :, 1], bins=nbins)
        ch3_hist = np.histogram(img[:, :, 2], bins=nbins)

        # Generating bin centers
        bin_edges = ch1_hist[1]
        bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2

        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))

        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    @staticmethod
    def visualize_hog_features(img, orientations=9, pixels_per_cell=8, cells_per_block=2, feature_vector=True):
        features, hog_image = hog(img, orientations=orientations,
                                  pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                                  cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        return features, hog_image

    @staticmethod
    def hog_features(img, hog_channel='ALL', orientations=9, pixels_per_cell=8, cells_per_block=2, feature_vector=True):
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                features = hog(img[:, :, channel], orientations=orientations,
                               pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                               cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=False,
                               visualise=False, feature_vector=feature_vector)
                hog_features.append(features)
            hog_features = np.ravel(hog_features)

        else:
            hog_features = hog(img[:, :, hog_channel], orientations=orientations,
                               pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                               cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=False,
                               visualise=False, feature_vector=feature_vector)
        return hog_features

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

        self.scaler = StandardScaler().fit(X)
        X = self.scaler.transform(X)

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

    def predict(self, features):
        scaled_features = self.scaler.transform(np.array(features).reshape(1, -1))
        return self.svc.predict(scaled_features)


class ImageSampler:
    def __init__(self, img_shape):

        # initialize windows, should be able to reuse as long as image shape is the same for each image
        windows = self.slide_window(img_shape, x_start_stop=[50, None], y_start_stop=[400, 500],
                                    xy_window=(64, 64), xy_overlap=(0.3, 0.3))
        windows += self.slide_window(img_shape, x_start_stop=[50, None], y_start_stop=[350, 600],
                                     xy_window=(128, 128), xy_overlap=(0.75, 0.75))
        windows += self.slide_window(img_shape, x_start_stop=[20, None], y_start_stop=[350, 700],
                                     xy_window=(175, 175), xy_overlap=(0.7, 0.7))


        self.windows = windows

    def sample_image(self, img, sample_shape=(64, 64)):
        samples = []
        for window in self.windows:
            samples_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], sample_shape)
            samples.append(samples_img)

        return samples, self.windows

    def slide_window(self, img_shape, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img_shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img_shape[0]

        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]

        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
        ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
        nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
        ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)

        window_list = []
        # Loop through window positions
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))

        return window_list
