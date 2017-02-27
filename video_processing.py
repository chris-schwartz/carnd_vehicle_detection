import numpy as np
from moviepy.editor import VideoFileClip
import cv2


class VideoProcessor:
    """
    This class generates uses a trained vehicle detection pipeline to detect images in an existing video and
    draws bounding boxes around each detected vehicle.
    """

    def __init__(self, trained_pipeline, frames_between_updates=1):
        self.pipeline = trained_pipeline
        self.last_labeled_boxes = []
        self.current_frame_number = 0
        self.frames_between_processing = frames_between_updates

    @staticmethod
    def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
        """ Draws given boxes on image """
        # Make a copy of the image
        imcopy = np.copy(img)
        count = 0

        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
            count += 1
        # Return the image copy with boxes drawn
        return imcopy

    def process_frame(self, frame):
        """ Draws boxes around detected cars for a single frame of video """

        # update labeled boxes every x frames
        if self.current_frame_number % self.frames_between_processing == 0:
            self.last_labeled_boxes = self.pipeline.detect_vehicles(frame, threshold=2)

        # draw boxes
        out_img = self.draw_boxes(frame, self.last_labeled_boxes)

        # keep track of frame count to know when the another frame needs to be processed
        self.current_frame_number += 1

        return out_img

    def process_video(self, src_path, dest_path):
        """
        This method detects vehicles in an existing video, draws bounding boxes around each detected vehicle and
        saves the output to a new video file.
        """
        # prevent pipeline from preventing stats when processing frame
        pipeline_verbosity = self.pipeline.verbose
        self.pipeline.set_verbose(False)

        input_video = VideoFileClip(src_path)
        out_clip = input_video.fl_image(self.process_frame)
        out_clip.write_videofile(dest_path, audio=False)

        # restore verbose flag for pipeline once finished
        self.pipeline.set_verbose(pipeline_verbosity)
