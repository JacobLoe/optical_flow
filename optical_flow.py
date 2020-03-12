import numpy as np
import cv2
import os
import argparse
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm
import shutil

STEP_SIZE = 300     # the steps in ms that are taken in a shot
DECIMALS = 2        # the number of decimals after rounding
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1]     #


def get_optical_flow(v_path, frame_width):

    vid = cv2.VideoCapture(v_path)
    summed_mags = []
    timestamp_ms = 0
    # iterate through all shots in a movie
    while vid.isOpened():
        # Capture frame-by-frame
        vid.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        ret, curr_frame = vid.read()  # if ret is false, frame has no content
        if not ret:
            break

        if timestamp_ms == 0:

            resolution_old = np.shape(curr_frame)
            ratio = resolution_old[1]/resolution_old[0]
            frame_height = int(frame_width/ratio)
            resolution_new = (frame_width, frame_height)
            prev_frame = cv2.resize(curr_frame, resolution_new)
            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:

            resolution_old = np.shape(curr_frame)
            ratio = resolution_old[1]/resolution_old[0]
            frame_height = int(frame_width/ratio)
            resolution_new = (frame_width, frame_height)
            curr_frame = cv2.resize(curr_frame, resolution_new)

            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # create the optical flow for two neighbouring frames
            flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # calculate the maximum possible magnitude
            max_mag = np.sqrt(np.power(np.shape(curr_frame)[0], 2) + np.power(np.shape(curr_frame)[1], 2))
            mag = mag / max_mag

            # sum up all the magnitudes between neighboring frames in the shot creating a single value
            # scale the value by the number of pixels in the image to keep the value between 0 and 1
            summed_mags.append(np.sum(mag) / (np.shape(mag)[0] * np.shape(mag)[1]))

            prev_frame = curr_frame     # save the current as the new previous frame for the next iteration
        timestamp_ms += STEP_SIZE

    vid.release()
    cv2.destroyAllWindows()

    # scale to a max-value of 1
    scaled_mags = summed_mags/np.max(summed_mags)
    # round the values to one significant digit
    rounded_mags = [np.round(x, decimals=DECIMALS) for x in scaled_mags]
    # digitize the values,
    indices_digitized_mags = np.digitize(rounded_mags, BINS)    # indices start a 1, npumpy array with shape of rounded_mags
    digitized_mags = [BINS[i-1] for i in indices_digitized_mags]    #

    return digitized_mags


def group_movie_magnitudes(digitized_mags):
    grouped_mags = []
    shot_timestamps = []
    prev_mag = 0
    start_ms = 0
    # iterate through the magnitudes,
    for i, mag in enumerate(digitized_mags):
        if i == 0:
            prev_mag = mag
        elif mag != prev_mag:
            grouped_mags.append(mag)
            end_ms = i*STEP_SIZE
            shot_timestamps.append((start_ms, end_ms))
            prev_mag = mag
            start_ms = end_ms
        else:
            pass
    return grouped_mags, shot_timestamps


def write_mag_to_csv(f_path, grouped_mags, shot_timestamps):
    if not os.path.isdir(os.path.join(f_path, 'optical_flow')):
        os.makedirs(os.path.join(f_path, 'optical_flow'))

    mag_csv_path = os.path.join(f_path, 'optical_flow/mag_optical_flow_{}.csv'.format(os.path.split(f_path)[1]))

    with open(mag_csv_path, 'w', newline='') as f:
        for i, mag in enumerate(grouped_mags):
            line = str(shot_timestamps[i][0])+' '+str(shot_timestamps[i][1])+' '+str(mag)
            f.write(line)
            f.write('\n')


def main(videos_path, features_path, frame_width):
    list_videos_path = glob.glob(os.path.join(videos_path, '**/*.mp4'), recursive=True)  # get the list of videos in videos_dir

    cp = os.path.commonprefix(list_videos_path)  # get the common dir between paths found with glob

    list_features_path = [os.path.join(
                         os.path.join(features_path,
                         os.path.relpath(p, cp))[:-4])
                         for p in list_videos_path]  # create a list of paths where all the data (shot-detection,frames,features) are saved to

    done = 0
    while done < len(list_videos_path):
        for v_path, f_path in tqdm(zip(list_videos_path, list_features_path), total=len(list_videos_path)):

            if not os.path.isdir(os.path.join(f_path, 'optical_flow')) and not os.path.isfile(os.path.join(f_path, 'optical_flow/.done')):
                print('optical flow is calculated for {}'.format(os.path.split(v_path)[1]))
                # os.makedirs(os.path.join(f_path, 'optical_flow'))
                digitized_mags = get_optical_flow(v_path, frame_width)
                # print(np.shape(digitized_mags))
                # print(summed_mags)
                grouped_mags, shot_timestamps = group_movie_magnitudes(digitized_mags)
                # print(np.shape(grouped_mags))
                # print(np.shape(shot_timestamps))
                # print(shot_timestamps)
                write_mag_to_csv(f_path, grouped_mags, shot_timestamps)
                open(os.path.join(f_path, 'optical_flow/.done'), 'a').close()
                done += 1
            elif os.path.isfile(os.path.join(f_path, 'optical_flow/.done')):    # do nothing if a .done-file exists
                done += 1
                print('optical flow was already done for {}'.format(os.path.split(v_path)[1]))
            elif os.path.isdir(os.path.join(f_path, 'optical_flow')) and not os.path.isfile(os.path.join(f_path, 'optical_flow/.done')):
                shutil.rmtree(os.path.join(f_path, 'optical_flow'))
                print('optical flow was not done correctly for {}'.format(os.path.split(v_path)[1]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("videos_dir", help="the directory where the video-files are stored")
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    parser.add_argument("--frame_width", type=int, default=129, help="set the width at which to which the frames are rescaled, default 129")
    # parser.add_argument('--ang_bins', type=list, default=[0, 45, 90, 135, 180, 225, 270, 315, 360], help='set the angle bins for the histogram, takes a list as input')
    # parser.add_argument('--mag_bins', type=list, default=[0, 20, 40, 60, 80, 100], help='set the magnitude bins for the histogram, takes a list as input')
    args = parser.parse_args()

    main(args.videos_dir, args.features_dir, args.frame_width)

