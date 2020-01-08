import numpy as np
import cv2
import os
import argparse
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm


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

    optical_flow_annotations = []

    vid = cv2.VideoCapture(v_path)
    for start_frame, end_frame in tqdm(shot_timestamps):

        # print(start_frame,end_frame)
        for i in range(int(end_frame / 1000 - start_frame / 1000) + 1):

            if i == 0 and not (start_frame / 1000 + i) == (int(end_frame / 1000 - start_frame / 1000) + 1):
                vid.set(cv2.CAP_PROP_POS_MSEC, start_frame + i * 1000)
                ret, frame = vid.read()
                prvs = cv2.resize(frame, video_resolution)
                prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)

            if not (start_frame / 1000 + i) == (int(end_frame / 1000 - start_frame / 1000) + 1):
                vid.set(cv2.CAP_PROP_POS_MSEC, start_frame + i * 1000)
                ret, frame = vid.read()
                next = cv2.resize(frame, video_resolution)
                next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                ang = ang * 180 / np.pi / 2

                ang_hist = np.histogram(ang, bins=ang_bins)
                map_hist = np.histogram(mag, bins=mag_bins)

                optical_flow_annotations.append(ang)

    vid.release()
    cv2.destroyAllWindows()
    return optical_flow_annotations


def main(videos_path, features_path, video_resolution, ang_bins, mag_bins):
    list_videos_path = glob.glob(os.path.join(videos_path, '**/*.mp4'), recursive=True)  # get the list of videos in videos_dir

    cp = os.path.commonprefix(list_videos_path)  # get the common dir between paths found with glob

    list_features_path = [os.path.join(
                         os.path.join(features_path,
                         os.path.relpath(p, cp))[:-4])  # add a new dir 'VIDEO_FILE_NAME/shot_detection' to the path
                         for p in list_videos_path]  # create a list of paths where all the data (shot-detection,frames,features) are saved to

    # print(list_videos_path[7])
    for v_path, f_path in zip(list_videos_path[7:], list_features_path[7:]):
        get_optical_flow(v_path, f_path, video_resolution, ang_bins, mag_bins)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("videos_dir", help="the directory where the video-files are stored")
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    args = parser.parse_args()

    video_resolution = (129, 129)
    ang_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    mag_bins = [0, 20, 40, 60, 80, 100]
    main(args.videos_dir, args.features_dir, video_resolution, ang_bins, mag_bins)

