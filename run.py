import time

from features import DataLoader
from features import FeatureExtractor
import numpy as np

if __name__ == '__main__':

    # load vehicle and non vehicle data
    loader = DataLoader()
    vehicle_data = loader.load_data(path="vehicles/**/*.png")
    non_vehicle_data = loader.load_data(path="non-vehicles/**/*.png")

    # color parameters
    bins = (16, 32, 64)
    sizes = ((32,32), (24,24), (16,16))
    colorspaces = ('RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb')

    # hog parameters
    orientation_configs = (6, 9, 12)
    pix_per_cells_configs = (4, 8, 12)
    cells_per_block_configs = (1,2,3)

    featureExtractor = FeatureExtractor()
    start = time.time()
    vehicle_features = featureExtractor.extract_features(vehicle_data,
                                                         colorspace=colorspaces[0],
                                                         spatial_size=sizes[0],
                                                         hist_bins=bins[0],
                                                         orientations=orientation_configs[1],
                                                         hog_pixels_per_cell=pix_per_cells_configs[1],
                                                         hog_cells_per_block=cells_per_block_configs[1],
                                                         use_hog=False)

    non_vehicle_features = featureExtractor.extract_features(non_vehicle_data,
                                                         colorspace=colorspaces[0],
                                                         spatial_size=sizes[0],
                                                         hist_bins=bins[0],
                                                         orientations=orientation_configs[1],
                                                         hog_pixels_per_cell=pix_per_cells_configs[1],
                                                         hog_cells_per_block=cells_per_block_configs[1],
                                                         use_hog=False)
    print("Finished extracting in", round(time.time() - start), "seconds")

    feature_array = np.asarray(vehicle_features)
    print("vehicle features:", feature_array.shape)
    print("Done")