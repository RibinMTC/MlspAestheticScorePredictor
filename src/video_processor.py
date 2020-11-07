import sys
from os.path import dirname, abspath

import cv2
import multiprocessing as mp
import numpy as np

from src.mlsp_model import MlspModel


def multi_processing_predict(group_number):
    cap = cv2.VideoCapture(test_video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_jump_unit * group_number)

    my_jump_unit = current_frame_jump_unit
    if group_number + 1 == num_processes:
        my_jump_unit = frame_count - current_frame_jump_unit * group_number
        print("My jump unit is: " + str(my_jump_unit))

    proc_frames = 0

    scores = []
    # try:
    while proc_frames < my_jump_unit:
        ret, frame = cap.read()
        if not ret:
            break

        score = float(mlsp_model.predict_from_frame(frame))
        scores.append(score)

        proc_frames += 1
    # except:
    #     cap.release()
    #     print("-----------Exception: " + str(sys.exc_info()[0]))

    cap.release()

    return scores


def predict_video_multi_process():
    print("Video processing using {} processes...".format(num_processes))

    p = mp.Pool(num_processes)
    results = p.map(multi_processing_predict, range(num_processes))
    results = np.hstack(results)
    p.close()
    p.join()
    print("Finished video multiprocessing")
    # print("Average: " + str(np.mean(results)))
    # print("Median: " + str(np.median(results)))


def get_total_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return total_frame_count


current_directory = dirname(dirname(abspath(__file__)))
ava_mlsp_root_path = current_directory + '/ava-mlsp/'

num_processes = 1 #mp.cpu_count()
test_video_path = '../resources/dance_video.mp4'
frame_count = get_total_frame_count(test_video_path)
current_frame_jump_unit = np.ceil(frame_count / num_processes)

mlsp_model = MlspModel(ava_mlsp_root_path)
predict_video_multi_process()


# from src.main_predictor import MainPredictor
#
# test_video_path = '../resources/inception_hallway_fight.mp4'
# num_of_frame = get_total_frame_count(test_video_path)
# print("Total number of frames to process: " + str(num_of_frame))
# main_predictor = MainPredictor()
# score = main_predictor.predict_score(test_video_path, 50, 100)
#
# print("Video avg score: " + str(score))


# import tensorflow as tf
# from ku import image_utils as img, applications as apps, model_helper as mh
# print(tf.__version__)
#
# current_directory = dirname(dirname(abspath(__file__)))
# ava_mlsp_root_path = current_directory + '/ava-mlsp/'
# mlsp_model = MlspModel(ava_mlsp_root_path)