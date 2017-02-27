import time
import cv2

from detection_components import DataLoader, SupportVectorClassifier, ImageSampler, HeatMapFilter
from detection_components import FeatureExtractor


class FeatureExtractionParams:
    """
    This class is used to provide a single place to define all the feature extraction parameters used throughout
    this project.
    """
    def __init__(self):
        self.colorspace = 'YCrCb'
        self.size = (32, 32)
        self.orientation = 12  # 9
        self.pix_per_cells = 8
        self.bin_count = 24  # 32
        self.cells_per_block = 2
        self.use_hog = True
        self.use_bin_spatial = True
        self.use_color_hist = True


class VehicleDetectionPipeline:
    """
    This class defines the main pipeline used to:
        - load training images
        - extract features from images
        - train a classifier
        - identify vehicles in an image using the trained classifer
    """

    def __init__(self, verbose=False):
        self.vehicle_features = []
        self.non_vehicle_features = []
        self.vehicle_data = []
        self.non_vehicle_data = []

        self.classifier = None
        self.verbose = verbose

        # feature extraction parameters
        params = FeatureExtractionParams()

        self.feature_extractor = FeatureExtractor(colorspace=params.colorspace,
                                                  spatial_size=params.size,
                                                  hist_bins=params.bin_count,
                                                  orientations=params.orientation,
                                                  hog_pixels_per_cell=params.pix_per_cells,
                                                  hog_cells_per_block=params.cells_per_block,
                                                  use_hog=params.use_hog,
                                                  use_color_hist=params.use_color_hist,
                                                  use_bin_spatial=params.use_bin_spatial)
        self.heatmap_filter = HeatMapFilter()

    def set_verbose(self, verbose):
        """Specify whether detailed information should be printed when running the pipeline."""
        self.verbose = verbose

    def load_training_data(self, vehicle_data_path, nonvehicle_data_path, use_pickle=True, reset_pickle=False):
        """Loads vehicle and non vehicle images to be used for training the classifier."""

        start = time.time()

        loader = DataLoader()
        self.vehicle_data = loader.load_data(path=vehicle_data_path, use_pickle=use_pickle, reset_pickle=reset_pickle)
        self.non_vehicle_data = loader.load_data(path=nonvehicle_data_path, use_pickle=use_pickle,
                                                 reset_pickle=reset_pickle)

        total = len(self.vehicle_data) + len(self.non_vehicle_data)
        if self.verbose:
            print("Finished loading {:d} images in {:0.1f} seconds.".format(total, time.time() - start))

        return self.vehicle_data, self.non_vehicle_data, total

    def extract_features(self):
        """ Extracts features for training from the loaded training data. """

        start = time.time()
        self.vehicle_features = self.feature_extractor.get_features(self.vehicle_data)
        self.non_vehicle_features = self.feature_extractor.get_features(self.non_vehicle_data)

        if self.verbose:
            print("Finished extracting features in", round(time.time() - start), "seconds")

    def train_classifier(self, test_size=0.25):
        """
        Trains a linear SVC for classifying vehicle images.  test_size indicates what percentage
        of data will be dedicated to testing the classifier.
        """

        start = time.time()
        self.classifier = SupportVectorClassifier(self.vehicle_features, self.non_vehicle_features)
        score = self.classifier.train_and_score(test_size=test_size)
        if self.verbose:
            print("Finished training in", round(time.time() - start),
                  "seconds with {:3.3f}% accuracy.".format(score * 100.0))

    def find_windows_with_vehicles(self, image, return_boxes=True, return_images=False):
        """
        Splits an image up into subsamples an runs each sample through a trained classifier
        to determine if the sample is a vehicle or not.

        Will return the box coordinates of each sample with a vehicle in it, an image with
        the positive sample boundaries drawn on them, or both based on the values of
        return_boxes and return_images.
        """

        start = time.time()
        car_images, boxes = [], []

        # get samples from image
        sampler = ImageSampler(img_shape=image.shape)
        samples, windows = sampler.sample_image(image)

        # get features for sample image
        samples_features = self.feature_extractor.get_features(samples)

        # determine if sample is a car image
        for idx, feature_sample in enumerate(samples_features):
            prediction = self.classifier.predict(feature_sample)

            # if car was detected in sample add to detected car images/boxes
            if prediction == 1:
                if return_images:
                    car_images.append(samples[idx])
                if return_boxes:
                    boxes.append(windows[idx])

        if self.verbose:
            count = max(len(boxes), len(car_images))
            print("Detected {:d} windows with vehicles in {:1.1f} seconds.".format(count, time.time() - start))

        if return_boxes and return_images:
            return car_images, boxes
        elif return_images:
            return car_images

        return boxes

    def detect_vehicles(self, image, threshold, return_heatmap=False, return_boxes=True):
        """
        Detects all vehicles in a given image. Threshold specifies the number of samples that must
        be exceeded before a sample is deemed a vehicle.  May return box coordinates for cars,
        a heat map, or both based on specified value for return_heatmap and return_boxes
        """

        boxes = self.find_windows_with_vehicles(image, return_boxes=True)
        self.heatmap_filter.update_heatmap(boxes, image.shape, threshold=threshold)

        if return_boxes and return_heatmap:
            return self.heatmap_filter.get_filtered_boxes(), self.heatmap_filter.get_heatmap()
        elif return_heatmap:
            return self.heatmap_filter.get_heatmap()

        return self.heatmap_filter.get_filtered_boxes()


if __name__ == '__main__':
    pipeline = VehicleDetectionPipeline()
    pipeline.load_training_data(vehicle_data_path="vehicles/**/*.png", nonvehicle_data_path="non-vehicles/**/*.png")
    pipeline.extract_features()
    pipeline.train_classifier(test_size=0.25)

    img = cv2.imread('project_video_images/frame62790.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    car_boxes = pipeline.detect_vehicles(img)
    print("Found", len(car_boxes), "cars.")
