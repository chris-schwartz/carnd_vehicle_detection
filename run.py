import time

from features import DataLoader, SupportVectorClassifier, ImageSampler
from features import FeatureExtractor
from multiprocessing import Process
import numpy as np


class VehicleDetectionPipline:
    def __init__(self):
        self.vehicle_features = []
        self.non_vehicle_features = []
        self.vehicle_data = []
        self.non_vehicle_data = []

        self.classifier = None

        # feature extraction parameters
        colorspace = 'YCrCb'
        size = (16, 16)
        orientation = 9
        pix_per_cells = 8
        bin_count = 32
        cells_per_block = 2
        use_hog = True
        use_bin_spatial = True
        use_color_hist = True

        self.feature_extractor = FeatureExtractor(colorspace=colorspace,
                                                  spatial_size=size,
                                                  hist_bins=bin_count,
                                                  orientations=orientation,
                                                  hog_pixels_per_cell=pix_per_cells,
                                                  hog_cells_per_block=cells_per_block,
                                                  use_hog=use_hog,
                                                  use_color_hist=use_color_hist,
                                                  use_bin_spatial=use_bin_spatial)

    def load_data(self, vehicle_data_path, nonvehicle_data_path):
        loader = DataLoader()
        self.vehicle_data = loader.load_data(path=vehicle_data_path)
        self.non_vehicle_data = loader.load_data(path=nonvehicle_data_path)

    def extract_features(self):
        start = time.time()
        self.vehicle_features = self.feature_extractor.get_features(self.vehicle_data)
        self.non_vehicle_features = self.feature_extractor.get_features(self.non_vehicle_data)

        print("Finished extracting features in", round(time.time() - start), "seconds")

    def train_classifier(self, test_size=0.25):
        start = time.time()
        self.classifier = SupportVectorClassifier(self.vehicle_features, self.non_vehicle_features)
        score = self.classifier.train_and_score(test_size=test_size)
        print("Finished training in", round(time.time() - start),
              "seconds with %{:3.3f} accuracy.".format(score * 100.0))

    def detect_vehicles_in_image(self, img):
        sampler = ImageSampler(img.shape)


if __name__ == '__main__':
    pipeline = VehicleDetectionPipline()
    pipeline.load_data(vehicle_data_path="vehicles/**/*.png", nonvehicle_data_path="non-vehicles/**/*.png")
    pipeline.extract_features(size=(16, 16), colorspace='YCrCb')
    pipeline.train_classifier(test_size=0.25)
