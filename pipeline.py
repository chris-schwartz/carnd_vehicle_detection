import time

import numpy as np

from detection_components import DataLoader, SupportVectorClassifier, ImageSampler, HeatMapFilter,FeatureExtractor, \
    PipelineParameters
from video_processing import VideoProcessor


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
        self.params = PipelineParameters()
        self.feature_extractor = FeatureExtractor(params=self.params)

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
            print("Vehicle Features Shape:", np.asarray(self.vehicle_features).shape)
            print("Non Vehicle Features Shape:", np.asarray(self.non_vehicle_features).shape)

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

    def detect_vehicles(self, image, return_heatmap=False, return_boxes=True):
        """
        Detects all vehicles in a given image. Threshold specifies the number of samples that must
        be exceeded before a sample is deemed a vehicle.  May return box coordinates for cars,
        a heat map, or both based on specified value for return_heatmap and return_boxes
        """

        boxes = self.find_windows_with_vehicles(image, return_boxes=True)
        self.heatmap_filter.update_heatmap(boxes, image.shape, threshold=self.params.filter_threshold,
                                           decay_rate=self.params.decay_rate)

        if return_boxes and return_heatmap:
            return self.heatmap_filter.get_filtered_boxes(), self.heatmap_filter.get_heatmap()
        elif return_heatmap:
            return self.heatmap_filter.get_heatmap()

        return self.heatmap_filter.get_filtered_boxes()


if __name__ == '__main__':
    pipeline = VehicleDetectionPipeline(verbose=True)

    # load up all training data
    pipeline.load_training_data(vehicle_data_path="vehicles/**/*.png", nonvehicle_data_path="non-vehicles/**/*.png")

    # extract features from loaded images
    pipeline.extract_features()

    # perform training
    pipeline.train_classifier(test_size=0.25)

    # apply pipeline to detect videos in short test video
    print("Using trained pipeline to generate new video.")
    video_processor = VideoProcessor(pipeline, frames_between_updates=2)
    video_processor.process_video("project_video.mp4", "longer_shot.mp4")
