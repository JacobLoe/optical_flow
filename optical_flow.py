import numpy as np
import cv2
import os
import argparse
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm
import shutil

FRAME_OFFSET_MS = 3*41  # frame offset in ms, one frame equals ~41ms, this jumps 3 frames ahead
STEP_SIZE = 1000     # the steps in ms that are taken in a shot

def read_shotdetect_xml(path):
    tree = ET.parse(path)
    root = tree.getroot().findall('content')
    timestamps = []
    for child in root[0].iter():
        if child.tag == 'shot':
            attribs = child.attrib
            timestamps.append((int(attribs['msbegin']), int(attribs['msbegin'])+int(attribs['msduration'])-1))  # ms
    return timestamps  # in ms


def get_optical_flow(v_path, shot_timestamps, frame_width, ang_bins, mag_bins):

    vid = cv2.VideoCapture(v_path)
    summed_shot_mags = []
    # iterate through all shots in a movie
    for start_ms, end_ms in tqdm(shot_timestamps, total=len(shot_timestamps)):
        # iterate through a shot, use a frame every STEP_SIZE milliseconds
        # if the shot is shorter than 2 seconds create a "fake" histogram between the first frame
        # in the shot and itself, the span of 2 seconds is dependent on the step size, a smaller step size leads to smaller span
        summed_mags = []
        if len(list(range(start_ms, end_ms, STEP_SIZE))) == 1:
            vid.set(cv2.CAP_PROP_POS_MSEC, start_ms)
            ret, frame = vid.read()

            resolution_old = np.shape(frame)
            ratio = resolution_old[1]/resolution_old[0]
            frame_height = int(frame_width/ratio)
            resolution_new = (frame_width, frame_height)

            frame = cv2.resize(frame, resolution_new)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # calculate the maximum possible magnitude
            max_mag = np.sqrt(np.power(np.shape(frame)[0], 2) + np.power(np.shape(frame)[1], 2))
            mag = mag / max_mag

            # sum up all the magnitudes between neighboring frames in the shot creating a single value
            # scale the value by the number of pixels in the image to keep the value between 0 and 1
            aux_summed_mags = np.sum(mag) / (np.shape(mag)[0] * np.shape(mag)[1])
            summed_mags.append(aux_summed_mags)

        else:
            first_frame = True
            for timestamp in range(start_ms, end_ms, STEP_SIZE):
                if not first_frame:
                    vid.set(cv2.CAP_PROP_POS_MSEC, timestamp)
                    ret, frame = vid.read()

                    resolution_old = np.shape(frame)
                    ratio = resolution_old[1] / resolution_old[0]
                    frame_height = int(frame_width / ratio)
                    resolution_new = (frame_width, frame_height)

                    curr_frame = cv2.resize(frame, resolution_new)
                    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

                    # create an image with the magnitudes and angles at each pixels
                    flow = cv2.calcOpticalFlowFarneback(prvs_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                    # calculate the maximum possible magnitude
                    max_mag = np.sqrt(np.power(np.shape(prvs_frame)[0], 2) + np.power(np.shape(prvs_frame)[1], 2))
                    mag = mag/max_mag

                    # sum up all the magnitudes between neighboring frames in the shot creating a single value
                    # scale the value by the number of pixels in the image to keep the value between 0 and 1
                    aux_summed_mags = np.sum(mag)/(np.shape(mag)[0]*np.shape(mag)[1])
                    summed_mags.append(aux_summed_mags)

                    prvs_frame = curr_frame  # save the current as the new previous frame for the next iteration

                if first_frame:
                    vid.set(cv2.CAP_PROP_POS_MSEC, timestamp)
                    ret, frame = vid.read()

                    resolution_old = np.shape(frame)
                    ratio = resolution_old[1] / resolution_old[0]
                    frame_height = int(frame_width / ratio)
                    resolution_new = (frame_width, frame_height)

                    prvs_frame = cv2.resize(frame, resolution_new)
                    prvs_frame = cv2.cvtColor(prvs_frame, cv2.COLOR_BGR2GRAY)
                    first_frame = False
        # sum up all the values in a shot and scale them by the number of values
        # print('len(summed_mags)', len(summed_mags), 'np.sum(summed_mags)', np.sum(summed_mags))
        aux_summed_shot_mags = np.sum(summed_mags)/len(summed_mags)
        summed_shot_mags.append(aux_summed_shot_mags)

    vid.release()
    cv2.destroyAllWindows()

    # scale
    summed_shot_mags = (summed_shot_mags-np.min(summed_shot_mags))/np.max(summed_shot_mags)
    # round the values to one significant digit
    summed_shot_mags = [np.round(x, decimals=1) for x in summed_shot_mags]

    return summed_shot_mags


def write_ang_and_mag_to_csv(f_path, shot_timestamps, summed_shot_mags):
    if not os.path.isdir(os.path.join(f_path, 'optical_flow')):
        os.makedirs(os.path.join(f_path, 'optical_flow'))

    mag_csv_path = os.path.join(f_path, 'optical_flow/mag_optical_flow_{}.csv'.format(os.path.split(f_path)[1]))
    with open(mag_csv_path, 'w', newline='') as f:
        for i, mag in enumerate(summed_shot_mags):
            line = str(shot_timestamps[i][0])+' '+str(shot_timestamps[i][1])+' '+str(mag)
            f.write(line)
            f.write('\n')

def main(videos_path, features_path, frame_width, ang_bins, mag_bins):
    list_videos_path = glob.glob(os.path.join(videos_path, '**/*.mp4'), recursive=True)  # get the list of videos in videos_dir

    cp = os.path.commonprefix(list_videos_path)  # get the common dir between paths found with glob

    list_features_path = [os.path.join(
                         os.path.join(features_path,
                         os.path.relpath(p, cp))[:-4])
                         for p in list_videos_path]  # create a list of paths where all the data (shot-detection,frames,features) are saved to

    done = 0
    while done < len(list_videos_path):
        for v_path, f_path in tqdm(zip(list_videos_path, list_features_path), total=len(list_videos_path)):

            shot_timestamps = read_shotdetect_xml(os.path.join(f_path, 'shot_detection/result.xml'))

            if not os.path.isdir(os.path.join(f_path, 'optical_flow')) and not os.path.isfile(os.path.join(f_path, 'optical_flow/.done')):
                summed_shot_mags = get_optical_flow(v_path, shot_timestamps, frame_width, ang_bins, mag_bins)

                write_ang_and_mag_to_csv(f_path, shot_timestamps, summed_shot_mags)
                open(os.path.join(f_path, 'optical_flow/.done'), 'a').close()
                done += 1
            elif os.path.isfile(os.path.join(f_path, 'optical_flow/.done')):    # do nothing if a .done-file exists
                done += 1
                print('image-extraction was already done for {}'.format(os.path.split(v_path)[1]))
            elif os.path.isdir(os.path.join(f_path, 'optical_flow')) and not os.path.isfile(os.path.join(f_path, 'optical_flow/.done')):
                shutil.rmtree(os.path.join(f_path, 'optical_flow'))
                print('image-extraction was not done correctly for {}'.format(os.path.split(v_path)[1]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("videos_dir", help="the directory where the video-files are stored")
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    parser.add_argument("--frame_width", type=int, default=129, help="set the width at which the frames are saved, default 129")
    parser.add_argument('--ang_bins', type=list, default=[0, 45, 90, 135, 180, 225, 270, 315, 360], help='set the angle bins for the histogram, takes a list as input')
    parser.add_argument('--mag_bins', type=list, default=[0, 20, 40, 60, 80, 100], help='set the magnitude bins for the histogram, takes a list as input')
    args = parser.parse_args()

    main(args.videos_dir, args.features_dir, args.frame_width, args.ang_bins, args.mag_bins)

