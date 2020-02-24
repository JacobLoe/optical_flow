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


def get_optical_flow(v_path, shot_timestamps, video_resolution, ang_bins, mag_bins):

    vid = cv2.VideoCapture(v_path)
    shot_mag_hists = []
    for i, ts in tqdm(enumerate(shot_timestamps), total=len(shot_timestamps)):
        ang_hists = []
        mag_hists = []
        start_ms, end_ms = ts
        # if the shot is shorter than 2 seconds create a "fake" histogram between the first frame
        # in the shot and itself, the span of 2 seconds is dependent on the step size, a smaller step size leads to smaller span
        # print(list(range(start_ms, end_ms, STEP_SIZE)) == [0], list(range(start_ms, end_ms, STEP_SIZE)), len(list(range(start_ms, end_ms, STEP_SIZE))))
        if len(list(range(start_ms, end_ms, STEP_SIZE))) == 1:
            vid.set(cv2.CAP_PROP_POS_MSEC, start_ms)
            ret, frame = vid.read()
            frame = cv2.resize(frame, video_resolution)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            ang = ang * 180 / np.pi  # convert the angles to degrees

            # calculate the maximum possible magnitude
            max_mag = np.sqrt(np.power(np.shape(frame)[0], 2) + np.power(np.shape(frame)[1], 2))

            # create histograms for the angles and magnitudes
            ang_hists.append(np.histogram(ang, bins=ang_bins)[0])
            mag_hists.append(np.histogram(mag, bins=mag_bins)[0])

        else:
            first_frame = True
            for timestamp in range(start_ms, end_ms, STEP_SIZE):
                if not first_frame:
                    vid.set(cv2.CAP_PROP_POS_MSEC, timestamp)
                    ret, frame = vid.read()
                    next_frame = cv2.resize(frame, video_resolution)
                    next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

                    # create an image with the magnitudes and angles at each pixels
                    flow = cv2.calcOpticalFlowFarneback(prvs_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    ang = ang * 180 / np.pi  # convert the angles to degrees

                    # calculate the maximum possible magnitude
                    max_mag = np.sqrt(np.power(np.shape(prvs_frame)[0], 2) + np.power(np.shape(prvs_frame)[1], 2))

                    # create histograms for the angles and magnitudes
                    ang_hists.append(np.histogram(ang, bins=ang_bins)[0])
                    mag_hists.append(np.histogram(mag, bins=mag_bins)[0])

                    prvs_frame = next_frame  #

                if first_frame:
                    vid.set(cv2.CAP_PROP_POS_MSEC, timestamp)
                    ret, frame = vid.read()
                    prvs_frame = cv2.resize(frame, video_resolution)
                    prvs_frame = cv2.cvtColor(prvs_frame, cv2.COLOR_BGR2GRAY)
                    first_frame = False

        # sum up the histograms in a shot, histograms show the magnitude of motion between two frames
        # print(mag_hists)
        for j, m in enumerate(mag_hists):
            if j == 0:
                summed_shot_mag_hist = m
            else:
                summed_shot_mag_hist += m
        # scale the new histogram by the number of histogram per shot
        # try:
        summed_shot_mag_hist = np.asarray(list(map(int, np.round(summed_shot_mag_hist/len(mag_hists)))))
        # except:
        #     print('summed_shot_mag_hist', summed_shot_mag_hist)
        #     print('mag_hists', mag_hists)
        #     print('len(mag_hists)', len(mag_hists))
        #     print('summed_shot_mag_hist/len(mag_hists)', summed_shot_mag_hist/len(mag_hists))
        #     print('np.round(summed_shot_mag_hist/len(mag_hists))', np.round(summed_shot_mag_hist/len(mag_hists)))
        # print('type summed_shot_list', type(summed_shot_mag_hist))
        # print(summed_shot_mag_hist, np.sum(summed_shot_mag_hist))
        shot_mag_hists.append(summed_shot_mag_hist)
        # break

    # # sum up the histogramms of the shots in a movie
    # for k, m in enumerate(shot_mag_hists):
    #     if k == 0:
    #         summed_movie_mag_hist = m
    #     else:
    #         summed_movie_mag_hist += m
    # # scale the new histogram by the number of shots in a movie
    # summed_movie_mag_hist = np.asarray(list(map(int, np.round(summed_movie_mag_hist/(i+1)))))

    # print('summed_shot_mag_hist: ', np.sum(summed_shot_mag_hist))
    # print('summed_movie_mag_hist: ', summed_movie_mag_hist, np.sum(summed_movie_mag_hist))
    vid.release()
    cv2.destroyAllWindows()
    return ang_hists, shot_mag_hists


def write_ang_and_mag_to_csv(f_path, shot_timestamps, ang_hist, mag_hist):
    if not os.path.isdir(os.path.join(f_path, 'optical_flow')):
        os.makedirs(os.path.join(f_path, 'optical_flow'))

    # ang_csv_path = os.path.join(f_path, 'optical_flow/ang_optical_flow.csv')
    # with open(ang_csv_path, 'w', newline='') as f:
    #     for i, ang in enumerate(ang_hist):
    #         line = str(shot_timestamps[i][0])+' '+str(shot_timestamps[i][1])+' '+"".join(str(list(ang))[1:-1].split())
    #         f.write(line)
    #         f.write('\n')

    mag_csv_path = os.path.join(f_path, 'optical_flow/mag_optical_flow.csv')
    with open(mag_csv_path, 'w', newline='') as f:
        for i, mag in enumerate(mag_hist):
            line = str(shot_timestamps[i][0])+' '+str(shot_timestamps[i][1])+' '+"".join(str(list(mag))[1:-1].split())
            f.write(line)
            f.write('\n')


def main(videos_path, features_path, video_resolution, ang_bins, mag_bins):
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
                ang_hist, mag_hist = get_optical_flow(v_path, shot_timestamps, video_resolution, ang_bins, mag_bins)

                write_ang_and_mag_to_csv(f_path, shot_timestamps, ang_hist, mag_hist)
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
    parser.add_argument('--video_resolution', type=tuple, default=(129, 129), help='set the resolution of the video, takes a tuple as input, default value is (129,129)')
    parser.add_argument('--ang_bins', type=list, default=[0, 45, 90, 135, 180, 225, 270, 315, 360], help='set the angle bins for the histogram, takes a list as input')
    parser.add_argument('--mag_bins', type=list, default=[0, 20, 40, 60, 80, 100], help='set the magnitude bins for the histogram, takes a list as input')
    args = parser.parse_args()

    main(args.videos_dir, args.features_dir, args.video_resolution, args.ang_bins, args.mag_bins)

