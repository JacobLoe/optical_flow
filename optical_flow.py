import numpy as np
import cv2
import os
import argparse
import glob
from tqdm import tqdm
import shutil
from scipy.spatial.distance import euclidean


BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]     #
ANGLE_BINS = [0, 45, 90, 135, 180, 225, 270, 315, 360]
VERSION = '20200609'      # the version of the script
aggregate = np.mean


def bin_values(value, bins):

    distances = []
    for b in bins:  # compute distance of the input to each bin
        distances.append(euclidean(b, value))

    i = np.argmin(distances)
    new_value = bins[i]
    return new_value


def resize_frame(frame, frame_width):
    resolution_old = np.shape(frame)
    ratio = resolution_old[1] / resolution_old[0]
    frame_height = int(frame_width / ratio)
    resolution_new = (frame_width, frame_height)
    resized_frame = cv2.resize(frame, resolution_new)
    return resized_frame


def read_frame(vid, timestamp, frame_width):
    vid.set(cv2.CAP_PROP_POS_FRAMES, timestamp)
    ret, frame = vid.read()  # if ret is false, frame has no content

    if not ret:
        return ret, 0

    if frame_width:  # reshape the frame according to the given frame width and the aspect ratio od the movie
        frame = resize_frame(frame, frame_width)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return ret, frame


def calculate_optical_flow(frame1, frame2):
    # create the optical flow for two neighbouring frames
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2,
                                        flow=None,
                                        pyr_scale=0.5,
                                        levels=3,
                                        winsize=15,
                                        iterations=3,
                                        poly_n=5,
                                        poly_sigma=1.2,
                                        flags=0)
    # mag and ang are matrices with the shape of the frame
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # sum up all the magnitudes between neighboring frames in the segment creating a single value
    summed_mags = np.sum(mag)

    return summed_mags


def get_optical_flow(v_path, frame_width):

    vid = cv2.VideoCapture(v_path)
    summed_mags = []    # list of summed up magnitudes
    timestamps = []         # list of timestamps, corresponding to summed_mags
    angles_histogram_list = []
    timestamp_frames = 0    # timestamp iterator, counted in frames
    step_size_in_frames = int(vid.get(cv2.CAP_PROP_FPS)*step_size/1000)  # convert the step_size and window_size from ms to frames, dependent on the fps of the movie
    window_size_in_frames = int(vid.get(cv2.CAP_PROP_FPS)*window_size/1000)

    timestamps.append((int(timestamp_frames/vid.get(cv2.CAP_PROP_FPS)*1000)))

    # go through the video and save the optical flow at each timestamp
    while vid.isOpened():
        # read the frame at the current timestamp, stop the reading if the video is finished
        ret, curr_frame = read_frame(vid, timestamp_frames, frame_width)
        if not ret:
            break
        # read the "future" frame, at the end of the window, stop the reading if the video is finished
        ret, future_frame = read_frame(vid, timestamp_frames + window_size_in_frames, frame_width)
        if not ret:
            break
        # calculate the optical flow for the current and the future frame, return the summed up magnitudes and a histogram for the angles found between the frames
        mag = calculate_optical_flow(curr_frame, future_frame)

        # save the two optical flow components
        summed_mags.append(mag)
        # move along the video according to the step_size
        timestamp_frames += step_size_in_frames

    timestamps.append((int(timestamp_frames/vid.get(cv2.CAP_PROP_FPS)*1000)))

    vid.release()
    cv2.destroyAllWindows()

    return summed_mags, timestamps


def aggregate_segments(summed_mags):
    aggregated_segments = []   # list of aggregated magnitudes

    ratio = window_size / step_size     # determines whether segments are aggregated or not, values <=1 mean no aggregation is happening (no segments are overlapping)
    segments_to_aggregate = int(ratio)     # determines how many segments are aggregated together
    if segments_to_aggregate == 1 and ratio > 1:   # if the ratio is greater than 1 there are segments to be aggregated, but the ratio would be rounded down to 1 in the next step
        segments_to_aggregate = 2                  # and prevent segment aggregation. To prevent this from happening segments_to_aggregate is set to 2

    if ratio <= 1:  # do nothing if the segments are not overlapping
        aggregated_segments = summed_mags
    else:
        for i in tqdm(range(len(summed_mags))):
            if i < (segments_to_aggregate-1):   # aggregate at timestamps where less segments then segments_to_aggregate where extracted
                #
                overlapping_segments = summed_mags[i:i+i+1]     # take the the number of overlapping segments at the timestamp (less than segments_to_aggregate
                aggregated_mags = aggregate(overlapping_segments)      # with the aggregate function, create one value
                aggregated_segments.append(aggregated_mags)

            # only aggregate segments if there are enough segments left in summed_mags
            else:
                overlapping_segments = summed_mags[i:i+segments_to_aggregate]     # get the overlapping segments from summed mags
                aggregated_mags = aggregate(overlapping_segments)      # with the aggregate function, create one value
                aggregated_segments.append(aggregated_mags)

                # start adding timestamps from the first timestamp where the max number of magnitudes overlap

    return aggregated_segments


def write_mag_to_csv(f_path, mag, segment_timestamps):
    if not os.path.isdir(os.path.join(f_path, 'optical_flow')):
        os.makedirs(os.path.join(f_path, 'optical_flow'))

    # scale magnitudes to 0-100
    max_mag = np.max(mag)
    mag = list(np.round(100*(mag/max_mag), decimals=2))

    mag_csv_path = os.path.join(f_path, 'optical_flow/mag_optical_flow_{}.csv'.format(os.path.split(f_path)[1]))

    with open(mag_csv_path, 'w', newline='') as f:
        mag = str(mag).strip('[').strip(']').replace(',', ' ')
        line = str(segment_timestamps[0]) + '\t' + str(segment_timestamps[1]) + '\t' + mag
        f.write(line)


def main(videos_path, features_path, frame_width):
    list_videos_path = glob.glob(os.path.join(videos_path, '**/*.mp4'), recursive=True)  # get the list of videos in videos_dir

    cp = os.path.commonprefix(list_videos_path)  # get the common dir between paths found with glob

    list_features_path = [os.path.join(
                         os.path.join(features_path,
                         os.path.relpath(p, cp))[:-4])
                         for p in list_videos_path]  # create a list of paths where all the data (segment-detection,frames,features) are saved to

    done = 0
    while done < len(list_videos_path):
        for v_path, f_path in tqdm(zip(list_videos_path, list_features_path), total=len(list_videos_path)):

            of_path = os.path.join(f_path, 'optical_flow')
            done_file_path = os.path.join(of_path, '.done')
            video_name = os.path.split(v_path)[1]

            if not os.path.isdir(of_path):
                print('optical flow is calculated for {}'.format(video_name))
                os.makedirs(of_path)

                print('get angles and magnitudes')
                summed_mags, timestamps = get_optical_flow(v_path, frame_width)

                print('aggregate segments')
                aggregated_segments = aggregate_segments(summed_mags)

                print('write results to csv')
                write_mag_to_csv(f_path, aggregated_segments, timestamps)

                # create a hidden file to signal that the optical flow for a movie is done
                # write the current version of the script in the file
                with open(done_file_path, 'a') as d:
                    d.write(VERSION)
                done += 1  # count the instances of the optical flow done correctly
            # do nothing if a .done-file exists and the versions in the file and the script match
            elif os.path.isfile(done_file_path) and open(done_file_path, 'r').read() == VERSION:
                done += 1  # count the instances of the optical flow done correctly
                print('optical flow was already done for {}'.format(video_name))
            # if the folder already exists but the .done-file doesn't, delete the folder
            elif os.path.isfile(done_file_path) and not open(done_file_path, 'r').read() == VERSION:
                shutil.rmtree(of_path)
                print('versions did not match for {}'.format(video_name))
            elif not os.path.isfile(done_file_path):
                shutil.rmtree(of_path)
                print('optical flow was not done correctly for {}'.format(video_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("videos_dir", help="the directory where the video-files are stored")
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    parser.add_argument("--frame_width", type=int, default=129, help="set the width at which to which the frames are rescaled, default is 129")
    parser.add_argument("--step_size", type=int, default=300, help="defines at which distances the optical flow is calculated, in milliseconds, default is 300")
    parser.add_argument("--window_size", type=int, default=300,
                        help="defines the range in which images for optical flow calculation are extracted, if window_size is equal to step_size two frames are extracted, default is 300")
    args = parser.parse_args()

    step_size = args.step_size
    window_size = args.window_size

    main(args.videos_dir, args.features_dir, args.frame_width)