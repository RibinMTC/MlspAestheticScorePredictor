import sys
from os.path import dirname, abspath

import cv2
from src.mlsp_model import MlspModel
from enum import Enum
import numpy as np
import mimetypes

# import multiprocessing as mp
# import subprocess as sp

mimetypes.init()


class ContentType(Enum):
    IMAGE = 1
    VIDEO = 2
    UNKNOWN = 3


class MainPredictor:
    def __init__(self):
        current_directory = dirname(dirname(abspath(__file__)))
        ava_mlsp_root_path = current_directory + '/ava-mlsp/'

        self.__mlsp_model = MlspModel(ava_mlsp_root_path)

        self.__content_type_based_predictor_dict = {ContentType.IMAGE: self.__predict_image,
                                                    ContentType.VIDEO: self.__predict_video}

    def predict_score(self, content_path, start_frame, end_frame):
        content_type = self.__get_content_type(content_path)
        if content_type == ContentType.UNKNOWN:
            print("The type for the following file is not supported: " + content_path)
            return -1

        score = self.__content_type_based_predictor_dict[content_type](content_path, start_frame, end_frame)
        return score

    def __get_content_type(self, content_path):
        mimestart = mimetypes.guess_type(content_path)[0]
        mimestart = mimestart.split('/')[0]
        if mimestart == 'video':
            return ContentType.VIDEO
        elif mimestart == 'image':
            return ContentType.IMAGE
        else:
            return ContentType.UNKNOWN

    def __predict_image(self, image_path, start_frame=0, end_frame=0):
        return self.__mlsp_model.predict(image_path)

    def __predict_video(self, video_path, start_frame, end_frame):
        cap = cv2.VideoCapture(video_path)
        scores = []
        average_score = -1
        median_score = -1
        try:
            for frameNumber in range(start_frame, end_frame + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)
                ret, frame = cap.read()
                if ret:
                    score = float(self.__mlsp_model.predict_from_frame(frame))
                    scores.append(score)
        except:
            print("Exception occurred in video: " + video_path)
            print(str(sys.exc_info()[0]))

        cap.release()

        num_of_scores = len(scores)
        if len(scores) > 0:
            assert(num_of_scores == (end_frame - start_frame + 1))
            average_score = np.mean(scores)
            median_score = np.median(scores)
            print("Average score: " + str(average_score))
            print("Median score: " + str(median_score))
            return average_score

        return average_score, median_score

    # def __predict_video(self, video_path):
    #     cap = cv2.VideoCapture(video_path)
    #     scores = []
    #     average_score = -1
    #     median_score = -1
    #     try:
    #         while cap.isOpened():
    #
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break
    #
    #             score = float(self.__mlsp_model.predict_from_frame(frame))
    #             scores.append(score)
    #     except:
    #         print("Exception occurred in video: " + video_path)
    #         print(str(sys.exc_info()[0]))
    #
    #     cap.release()
    #
    #     if len(scores) > 0:
    #         average_score = np.mean(scores)
    #         median_score = np.median(scores)
    #         print("Average score: " + str(average_score))
    #         print("Median score: " + str(median_score))
    #         return average_score
    #
    #     return average_score, median_score

    # def __get_total_frame_count(self, video_path):
    #     cap = cv2.VideoCapture(video_path)
    #     total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     return total_frame_count
    #
    # def __multi_processing_predict(self, group_number):
    #     cap = cv2.VideoCapture(self.__current_video_path)
    #
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, self.__current_frame_jump_unit * group_number)
    #
    #     proc_frames = 0
    #
    #     scores = []
    #     average_score = -1
    #     median_score = -1
    #
    #     try:
    #         while proc_frames < self.__current_frame_jump_unit:
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break
    #
    #             score = float(self.__mlsp_model.predict_from_frame(frame))
    #             scores.append(score)
    #
    #             proc_frames += 1
    #     except:
    #         cap.release()
    #
    #     cap.release()
    #
    #     return scores
    #
    # def __predict_video_multi_process(self, video_path):
    #     num_processes = mp.cpu_count()
    #     frame_count = self.__get_total_frame_count(video_path)
    #     self.__current_video_path = video_path
    #     self.__current_frame_jump_unit = frame_count // num_processes
    #     print("Video processing using {} processes...".format(num_processes))
    #
    #     p = mp.Pool(num_processes)
    #     results = p.map(self.__multi_processing_predict, range(num_processes))
    #     p.close()
    #     p.join()
    #     print("Finished video multiprocessing")
