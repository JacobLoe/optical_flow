import numpy as np
import cv2
import os
import argparse
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm
import csv


def read_shotdetect_xml(path):
    tree = ET.parse(path)
    root = tree.getroot().findall('content')
    timestamps = []
    for child in root[0].iter():
        if child.tag == 'shot':
            items = child.items()
            timestamps.append((int(items[4][1]), int(items[4][1])+int(items[2][1])-1))  # ms
    return timestamps  # in ms


def get_optical_flow(v_path, f_path, video_resolution, ang_bins, mag_bins):
    shot_timestamps = read_shotdetect_xml(os.path.join(f_path, 'shot_detection/result.xml'))

    ang_hist = []
    mag_hist = []
    vid = cv2.VideoCapture(v_path)
    for start_frame_ms, end_frame_ms in tqdm(shot_timestamps):

        vid.set(cv2.CAP_PROP_POS_MSEC, start_frame_ms)
        ret, frame = vid.read()
        prvs = cv2.resize(frame, video_resolution)
        prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)

        offset = int(end_frame_ms / 1000 - start_frame_ms / 1000)     # calculate an offset to the start_frame to get the last frame of the shot
        vid.set(cv2.CAP_PROP_POS_MSEC, start_frame_ms + offset*1000)
        ret, frame = vid.read()
        next = cv2.resize(frame, video_resolution)
        next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

        # create an image with the magnitudes and angles at each pixels
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        ang = ang * 180 / np.pi / 2     # convert the angles to degrees

        # create histograms for the angles and magnitudes
        ang_hist.append(np.histogram(ang, bins=ang_bins)[0])
        mag_hist.append(np.histogram(mag, bins=mag_bins)[0])

    vid.release()
    cv2.destroyAllWindows()
    return ang_hist, mag_hist


def write_to_csv(f_path, ang_hist, mag_hist):

    shot_timestamps = read_shotdetect_xml(os.path.join(f_path, 'shot_detection/result.xml'))

    for i, st in enumerate(shot_timestamps):
        start_frame_ms, end_frame_ms = st
        csv_path = os.path.join(f_path, 'optical_flow/'+str(start_frame_ms)+'.csv')
        break
    # print(csv_path)


def main(videos_path, features_path, video_resolution, ang_bins, mag_bins):
    list_videos_path = glob.glob(os.path.join(videos_path, '**/*.mp4'), recursive=True)  # get the list of videos in videos_dir

    cp = os.path.commonprefix(list_videos_path)  # get the common dir between paths found with glob

    list_features_path = [os.path.join(
                         os.path.join(features_path,
                         os.path.relpath(p, cp))[:-4])  # add a new dir 'VIDEO_FILE_NAME/shot_detection' to the path
                         for p in list_videos_path]  # create a list of paths where all the data (shot-detection,frames,features) are saved to

    # print(list_videos_path[7])
    for v_path, f_path in zip(list_videos_path[7:], list_features_path[7:]):
    # for v_path, f_path in zip(list_videos_path, list_features_path):
        ang_hist, mag_hist = get_optical_flow(v_path, f_path, video_resolution, ang_bins, mag_bins)
        write_to_csv(f_path, ang_hist, mag_hist)
        break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("videos_dir", help="the directory where the video-files are stored")
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    parser.add_argument('--video_resolution', type=tuple, default=(129, 129), help='set the resolution of the video, takes a tuple as input, default value is (129,129)')
    parser.add_argument('--ang_bins', type=list, default=[0, 45, 90, 135, 180, 225, 270, 315, 360], help='set the angle bins for the histogram, takes a list as input')
    parser.add_argument('--mag_bins', type=list, default=[0, 20, 40, 60, 80, 100], help='set the magnitude bins for the histogram, takes a list as input')
    args = parser.parse_args()

    main(args.videos_dir, args.features_dir, args.video_resolution, args.ang_bins, args.mag_bins)

