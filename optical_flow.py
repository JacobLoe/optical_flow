import numpy as np
import cv2
import os
import argparse
import glob
from tqdm import tqdm
import shutil

STEP_SIZE = 1000     # the steps a video is
WINDOW_SIZE = 2000  #
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1]     #
ANGLE_BINS = [0, 45, 90, 135, 180, 225, 270, 315, 360]
VERSION = '2020428'      # the version of the script
aggregate = np.mean


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

    # sum up all the magnitudes between neighboring frames in the shot creating a single value
    summed_mags = np.sum(mag)

    # digitize the angles
    ang = ang * 180 / np.pi
    angles_newshape = np.shape(ang)[0] * np.shape(ang)[1]
    ang = np.reshape(ang, newshape=angles_newshape)
    indices_digitized_angles = np.digitize(ang, ANGLE_BINS, right=True)
    digitized_angles = [ANGLE_BINS[i] for i in indices_digitized_angles]

    # create an empty histogram
    angles_histogram = {b: [0, 0] for b in ANGLE_BINS}
    # flatten the magnitudes to the same shape as the digitized_angles
    flattened_mag = np.reshape(mag, newshape=np.shape(mag)[0] * np.shape(mag)[1])
    for i, angle in enumerate(digitized_angles):
        for b in ANGLE_BINS:
            if b == angle:
                angles_histogram[b][0] += 1
                angles_histogram[b][1] += flattened_mag[i]

    return summed_mags, angles_histogram


def get_optical_flows(v_path, frame_width):

    vid = cv2.VideoCapture(v_path)
    summed_mags = []    # list of summed up magnitudes
    timestamps = []         # list of timestamps, corresponding to summed_mags
    angles_histogram_list = []
    timestamp_frames = 0    # timestamp iterator, counted in frames
    step_size_in_frames = int(vid.get(cv2.CAP_PROP_FPS)*STEP_SIZE/1000)  # convert the STEP_SIZE and WINDOW_SIZE from ms to frames, dependent on the fps of the movie
    window_size_in_frames = int(vid.get(cv2.CAP_PROP_FPS)*WINDOW_SIZE/1000)

    ratio = WINDOW_SIZE / STEP_SIZE    # determine how many steps fit in a window
    frames_in_a_window = int(ratio)  # determine how many frames are used for optical flow extraction
    if frames_in_a_window == 1:   # optical flow needs two frames to be calculated
        frames_in_a_window = 1

    print('frames in a window', frames_in_a_window)
    print('window size in frames: ', window_size_in_frames)
    #
    # print(sdag)
    # go through the video and save the optical flow at each timestamp
    while vid.isOpened():
        # read frames in a window and save them
        frames = []
        for i in range(frames_in_a_window):
            # create equally spaced timestamps
            ts = timestamp_frames + (i * window_size_in_frames / frames_in_a_window)
            print(ts)
            ret, frame = read_frame(vid, ts, frame_width)
            if not ret:
                break
            frames.append(frame)
        if not ret:
            break

        print('shape frames', np.shape(frames))
        prev_frame = frames[0]

        for f in frames[1:]:
            mag, angles_histogram = calculate_optical_flow(prev_frame, f)
            prev_frame = f

        print(asd)
        # # read the frame at the current timestamp, stop the reading if the video is finished
        # ret, curr_frame = read_frame(vid, timestamp_frames, frame_width)
        # if not ret:
        #     break
        # # read the "future" frame, at the end of the window, stop the reading if the video is finished
        # ret, future_frame = read_frame(vid, timestamp_frames + window_size_in_frames, frame_width)
        # if not ret:
        #     break
        # # calculate the optical flow for the current and the future frame, return unchanged magnitudes and a histogram for the angles found between the frames
        # mag, angles_histogram = calculate_optical_flow(curr_frame, future_frame)

        # save the timestamp of the current frame
        timestamps.append(int(timestamp_frames/vid.get(cv2.CAP_PROP_FPS)*1000))
        # save the two optical flow components
        summed_mags.append(mag)
        angles_histogram_list.append(angles_histogram)

        # move along the video according to the STEP_SIZE
        timestamp_frames += step_size_in_frames
    vid.release()
    cv2.destroyAllWindows()
    return summed_mags, angles_histogram_list, timestamps


def aggregate_shots(summed_mags, timestamps):
    aggregated_shots = []   # list of aggregated magnitudes
    aggregated_timestamps = []  # list of timestamps, corresponding to aggregated_shots
    #
    ratio = WINDOW_SIZE / STEP_SIZE     # determines whether shots are aggregated or not
    shots_to_aggregate = int(ratio)     # determines how many shots are aggregated together
    if shots_to_aggregate == 1 and ratio > 1:   # if the ratio is greater than 1 there are shots to be aggregated, but the ratio would be rounded down to 1 in the next step
        shots_to_aggregate = 2                  # and prevent shot aggregation. To prevent this from happening shots_to_aggregate is set to 2, if
    # print('ratio window/step: ', ratio)
    # print('shots_to_aggregate: ', shots_to_aggregate)
    # check the ratio of WINDOW_SIZE and STEP_SIZE, if its below or equal to 1 there is no overlap between shots and aggregation is not needed
    if not ratio <= 1:
        for i in tqdm(range(len(summed_mags))):
            # only aggregate shots if there are enough shots left in summed_mags
            if i+shots_to_aggregate <= len(summed_mags):
                overlapping_shots = summed_mags[i:i+shots_to_aggregate]
                # print(overlapping_shots, np.shape(overlapping_shots), i+shots_to_aggregate, len(summed_mags))
                aggregated_mags = aggregate(overlapping_shots)
                aggregated_shots.append(aggregated_mags)
                # start adding timestamps from the first timestamp where the max number of magnitudes overlap
                aggregated_timestamps.append(timestamps[i+shots_to_aggregate-1])
            # else:
            #     overlapping_shots = summed_mags[i:i+shots_to_aggregate]
            #     print(overlapping_shots, np.shape(overlapping_shots), i+shots_to_aggregate, len(summed_mags))
    else:
        aggregated_shots = summed_mags
        aggregated_timestamps = timestamps

    #
    _, magnitude_bins = np.histogram(aggregated_shots, bins=10000)

    # print(summed_mags,'\n')
    # print(timestamps,'\n')
    #
    # print('shape timestamps', np.shape(timestamps))
    # print('shape summed mags', np.shape(summed_mags))
    # # print(np.shape(angles_histogram_list))
    # print('shape aggregated shots', np.shape(aggregated_shots))
    # print('shape aggregated timestamps', np.shape(aggregated_timestamps))
    #
    # print(aggregated_shots)
    # print(aggregated_timestamps)

    return aggregated_shots, magnitude_bins, aggregated_timestamps


def group_angles_and_magnitudes(summed_mags, angles_histogram_list, timestamps):
    grouped_mags = [summed_mags[0]]  # start the grouped mag list
    grouped_angles = [angles_histogram_list[0]]  # save the first the first histogram
    shot_timestamps = []
    prev_mag = summed_mags[0]  # save the first magnitude for later comparison as previous magnitude
    start_ms = 0
    end_ms = timestamps[0]  # set the end timestamp
    j = 0
    # iterate through the magnitudes,
    for i, curr_mag in enumerate(summed_mags[1:]):
        if prev_mag == curr_mag:  # as long as the previous and current magnitude are the same
            end_ms = timestamps[i]     # set the end of the "current" shot to the timestamp of the current magnitude

            # sum up all the histograms (of the angles) and the magnitudes for each angle
            for key in grouped_angles[j]:
                grouped_angles[j][key][0] += angles_histogram_list[i][key][0]
                grouped_angles[j][key][1] += angles_histogram_list[i][key][1]

        else:   # if the magnitudes deviate add a new entry
            shot_timestamps.append((start_ms, end_ms))  # set the shot boundaries
            start_ms = end_ms   # set the start timestamp of the next shot as the end of the current shot
            end_ms = timestamps[i]  # set the end of the "current" shot to the timestamp of the current magnitude

            grouped_mags.append(curr_mag)  # add the next value to the grouped mag list
            prev_mag = curr_mag  # save the first magnitude for later comparison as previous mag

            grouped_angles.append(angles_histogram_list[i])
            j += 1

    shot_timestamps.append((start_ms, end_ms))  # set the shot boundaries for the last shot

    return grouped_mags, grouped_angles, shot_timestamps


def find_max_magnitude(grouped_mags, magnitude_bins, grouped_angles, shot_timestamps):

    # go through the bins in reverse
    # for each magnitude see if
    reversed_magnitude_bins = list(reversed(magnitude_bins))
    nmm = False
    new_max_magnitude = 0
    for i, b in enumerate(reversed_magnitude_bins):
        if i == 0:
            for j, m in enumerate(grouped_mags):
                # if a shot that falls into the bin >=b is longer than the STEP_SIZE stop the search
                # and take the magnitude of this shot as the new max magnitude for scaling
                if m >= b and shot_timestamps[j][1]-shot_timestamps[j][0] > STEP_SIZE:
                    new_max_magnitude = m
                    nmm = True
                    break
        else:
            # for each shot
            for j, m in enumerate(grouped_mags):
                # check if m is higher/equal than the current bin and lower than the previous
                if b <= m < reversed_magnitude_bins[i-1] and shot_timestamps[j][1]-shot_timestamps[j][0] > STEP_SIZE:
                    new_max_magnitude = m
                    nmm = True
                    break
        if nmm:
            break

    # scale the magnitudes using the new found "maximum" magnitude
    # if no magnitude was found (no shot is longer than the STEP_SIZE), take the maximum magnitude of the movie as the factor
    if new_max_magnitude == 0:
        new_max_magnitude = np.max(grouped_mags)
        scaled_mags = grouped_mags / new_max_magnitude
        print('no new maximum magnitude could be found, ')
    else:
        scaled_mags = grouped_mags / new_max_magnitude

    # scale the magnitudes that are corresponding to the angles
    for i, al in enumerate(grouped_angles):
        for key_angles in al:
            grouped_angles[i][key_angles][1] / new_max_magnitude
    # print(np.min(scaled_mags), np.max(scaled_mags))

    # clip values greater than 1 to 1
    clipped_mags = np.clip(scaled_mags, 0, 1)

    indices_digitized_mags = np.digitize(clipped_mags, BINS, right=True)    # returns a numpy array with shape of rounded_mags, the indices correspond to the position in BINS
    digitized_mags = [BINS[i] for i in indices_digitized_mags]    # map the magnitudes to the values in BINS

    timestamps = [end for begin, end in shot_timestamps]

    return digitized_mags, grouped_angles, timestamps


def find_dominant_movement(grouped_angles):
    dominant_angle_per_shot = []
    meta_info = []
    for i, histogram in enumerate(grouped_angles):
        # print(histogram)
        aux_histogram = {k:v[0]*v[1] for k, v in histogram.items()}     # multiply the magnitude and angle count to get a measure of movement for each angle
        hist_mean = np.mean(list(aux_histogram.values()))
        hist_var = np.var(list(aux_histogram.values()))
        # print(hist_var, hist_mean)
        aux_histogram = {k:((v-hist_mean)/hist_var if hist_var!=0 else 0) for k,v in aux_histogram.items()}     # normalize the histtogram
        # print(aux_histogram)
        # print(np.mean(list(aux_histogram.values())), np.var(list(aux_histogram.values())))
        # print(np.min(list(aux_histogram.values())),np.max(list(aux_histogram.values())))
        # if i==1:
        #     break

        reversed_histogram = {v:k for k, v in aux_histogram.items()}   # reverse the histogramm to allow it to be searched by the value
        dominant = sorted(aux_histogram.values(), reverse=True)[0]   # sort the histogram by magnittude times angle, descending, return the biggest value
        dominant_angle_per_shot.append(reversed_histogram[dominant])   # return the angle corresponding to the value

        # add additional information about the shot, whole histogram, min/max/average-values, variance
        # print(hist_values, type(hist_values))
        # print('np min max', np.min(hist_values), np.max(hist_values))
        # print('numpy mean', np.mean(hist_values))
        # print('numpy variance', np.var(hist_values, dtype='float64'))
        # print(asdada)
        info = [np.min(list(aux_histogram.values())), np.max(list(aux_histogram.values())),
                np.mean(list(aux_histogram.values())), np.var(list(aux_histogram.values()))]
        meta_info.append(info)
    return dominant_angle_per_shot, meta_info


def write_mag_to_csv(f_path, grouped_mags, shot_timestamps):
    if not os.path.isdir(os.path.join(f_path, 'optical_flow')):
        os.makedirs(os.path.join(f_path, 'optical_flow'))

    mag_csv_path = os.path.join(f_path, 'optical_flow/mag_optical_flow_{}.csv'.format(os.path.split(f_path)[1]))

    with open(mag_csv_path, 'w', newline='') as f:
        for i, mag in enumerate(grouped_mags):
            line = str(shot_timestamps[i][0])+' '+str(shot_timestamps[i][1])+' '+str(mag)
            f.write(line)
            f.write('\n')


def write_angle_to_csv(f_path, dominant_angle_per_shot, angle_meta_info,shot_timestamps):
    if not os.path.isdir(os.path.join(f_path, 'optical_flow')):
        os.makedirs(os.path.join(f_path, 'optical_flow'))

    mag_csv_path = os.path.join(f_path, 'optical_flow/angle_optical_flow_{}.csv'.format(os.path.split(f_path)[1]))

    with open(mag_csv_path, 'w', newline='') as f:
        for i, angle in enumerate(dominant_angle_per_shot):
            ami = str(angle_meta_info[i]).replace(' ', '')
            line = str(shot_timestamps[i][0])+' '+str(shot_timestamps[i][1])+' '+str(angle)+' '+ami
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

            of_path = os.path.join(f_path, 'optical_flow')
            done_file_path = os.path.join(of_path, '.done')
            video_name = os.path.split(v_path)[1]

            if not os.path.isdir(of_path):
                print('optical flow is calculated for {}'.format(video_name))
                os.makedirs(of_path)

                print('get angles and magnitudes')
                summed_mags, angles_histogram_list, timestamps = get_optical_flows(v_path, frame_width)
                # print('summed mags, angles, timestamps: ', np.shape(summed_mags), np.shape(angles_histogram_list), np.shape(timestamps))

                print('aggregate shots')
                aggregated_shots, magnitude_bins, aggregated_timestamps = aggregate_shots(summed_mags, timestamps)

                print('group angles and manitudes into shots')
                grouped_mags, grouped_angles, shot_timestamps = group_angles_and_magnitudes(aggregated_shots, angles_histogram_list, aggregated_timestamps)
                # print('mags, angles, timestamps: ', np.shape(grouped_mags), np.shape(grouped_angles), np.shape(shot_timestamps))

                print('find max magnitude')
                digitized_mags, new_grouped_angles, new_timestamps = find_max_magnitude(grouped_mags, magnitude_bins, grouped_angles, shot_timestamps)

                print('group angles and manitudes into shots with scaled mags')
                grouped_mags, grouped_angles, shot_timestamps = group_angles_and_magnitudes(digitized_mags, new_grouped_angles, new_timestamps)

                print('find dominant movements')
                dominant_angle_per_shot, angle_meta_info = find_dominant_movement(grouped_angles)

                print('write results to csv')
                write_angle_to_csv(f_path, dominant_angle_per_shot, angle_meta_info, shot_timestamps)
                write_mag_to_csv(f_path, grouped_mags, shot_timestamps)

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
    parser.add_argument("--frame_width", type=int, default=129, help="set the width at which to which the frames are rescaled, default 129")
    args = parser.parse_args()

    main(args.videos_dir, args.features_dir, args.frame_width)
