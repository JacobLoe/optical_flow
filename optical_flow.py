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
VERSION = '20200602'      # the version of the script
aggregate = np.mean


def bin_values(value, bins):

    distances = []
    for b in bins:  # compute distance of the input to each bin
        distances.append(euclidean(b, value))

    i = np.argmin(distances)    #
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

    ang = ang * 180 / np.pi     #

    angles_newshape = np.shape(ang)[0] * np.shape(ang)[1]   # reshape the angles to be one-dimensional
    ang = np.reshape(ang, newshape=angles_newshape)

    # digitize the angles
    # indices_digitized_angles = np.digitize(ang, ANGLE_BINS, right=True)
    # digitized_angles = [ANGLE_BINS[i] for i in indices_digitized_angles]
    digitized_angles = [bin_values(a, ANGLE_BINS) for a in ang]

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


def get_optical_flow(v_path, frame_width):

    vid = cv2.VideoCapture(v_path)
    summed_mags = []    # list of summed up magnitudes
    timestamps = []         # list of timestamps, corresponding to summed_mags
    angles_histogram_list = []
    timestamp_frames = 0    # timestamp iterator, counted in frames
    step_size_in_frames = int(vid.get(cv2.CAP_PROP_FPS)*step_size/1000)  # convert the step_size and window_size from ms to frames, dependent on the fps of the movie
    window_size_in_frames = int(vid.get(cv2.CAP_PROP_FPS)*window_size/1000)

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
        mag, angles_histogram = calculate_optical_flow(curr_frame, future_frame)

        # save the "segment boundaries" of the current frame, begin is the current timestamp, end is current+step_size
        timestamps.append((int(timestamp_frames/vid.get(cv2.CAP_PROP_FPS)*1000),
                           int((timestamp_frames+step_size_in_frames)/vid.get(cv2.CAP_PROP_FPS)*1000)))
        # save the two optical flow components
        summed_mags.append(mag)
        angles_histogram_list.append(angles_histogram)

        # move along the video according to the step_size
        timestamp_frames += step_size_in_frames
    vid.release()
    cv2.destroyAllWindows()

    assert np.shape(summed_mags)[0] == np.shape(angles_histogram_list)[0]
    assert np.shape(summed_mags)[0] == np.shape(timestamps)[0]

    return summed_mags, angles_histogram_list, timestamps


def aggregate_segments(summed_mags, timestamps):
    aggregated_segments = []   # list of aggregated magnitudes
    aggregated_timestamps = []  # list of timestamps, corresponding to aggregated_segments

    ratio = window_size / step_size     # determines whether segments are aggregated or not, values <=1 mean no aggregation is happening (no segments are overlapping)
    segments_to_aggregate = int(ratio)     # determines how many segments are aggregated together
    if segments_to_aggregate == 1 and ratio > 1:   # if the ratio is greater than 1 there are segments to be aggregated, but the ratio would be rounded down to 1 in the next step
        segments_to_aggregate = 2                  # and prevent segment aggregation. To prevent this from happening segments_to_aggregate is set to 2

    if ratio <= 1:  # do nothing if the segments are not overlapping
        aggregated_segments = summed_mags
        aggregated_timestamps = timestamps
    else:
        for i in tqdm(range(len(summed_mags))):
            if i < (segments_to_aggregate-1):   # aggregate at timestamps where less segments then segments_to_aggregate where extracted
                #
                overlapping_segments = summed_mags[i:i+i+1]     # take the the number of overlapping segments at the timestamp (less than segments_to_aggregate
                aggregated_mags = aggregate(overlapping_segments)      # with the aggregate function, create one value
                aggregated_segments.append(aggregated_mags)

                aggregated_timestamps.append(timestamps[i])
            # only aggregate segments if there are enough segments left in summed_mags
            else:
                overlapping_segments = summed_mags[i:i+segments_to_aggregate]     # get the overlapping segments from summed mags
                aggregated_mags = aggregate(overlapping_segments)      # with the aggregate function, create one value
                aggregated_segments.append(aggregated_mags)

                # start adding timestamps from the first timestamp where the max number of magnitudes overlap
                aggregated_timestamps.append(timestamps[i])

    # compute bins from the aggregated segments to be used for the scaling of the magnitudes
    _, magnitude_bins = np.histogram(aggregated_segments, bins=10000)

    assert len(summed_mags) == len(aggregated_timestamps)
    assert len(aggregated_timestamps) == len(aggregated_segments)

    return aggregated_segments, magnitude_bins, aggregated_timestamps


def group_angles_and_magnitudes(summed_mags, angles_histogram_list, timestamps):

    grouped_mags = [summed_mags[0]]  # start the grouped mag list
    grouped_angles = [angles_histogram_list[0]]  # save the first the first histogram
    segment_timestamps = []
    prev_mag = summed_mags[0]  # save the first magnitude for later comparison as previous magnitude
    start_ms = timestamps[0][0]
    end_ms = timestamps[0][1]  # set the end timestamp
    j = 0
    # iterate through the magnitudes, timestamps and angles starting from the second element
    for curr_mag, curr_ts, curr_angles_histogram in zip(summed_mags[1:], timestamps[1:], angles_histogram_list[1:]):
        if prev_mag == curr_mag:  # as long as the previous and current magnitude are the same
            end_ms = curr_ts[1]     # set the end of the "current" segment to the timestamp of the current magnitude

            # sum up all the histograms (of the angles) and the magnitudes for each angle
            for key in grouped_angles[j]:
                grouped_angles[j][key][0] += curr_angles_histogram[key][0]
                grouped_angles[j][key][1] += curr_angles_histogram[key][1]

        else:   # if the magnitudes deviate add a new entry
            segment_timestamps.append((start_ms, end_ms))  # set the segment boundaries
            start_ms = end_ms   # set the start timestamp of the next segment as the end of the current segment
            end_ms = curr_ts[1]  # set the end of the "current" segment to the timestamp of the current magnitude

            grouped_mags.append(curr_mag)  # add the next value to the grouped mag list
            prev_mag = curr_mag  # save the first magnitude for later comparison as previous mag

            grouped_angles.append(curr_angles_histogram)
            j += 1

    segment_timestamps.append((start_ms, end_ms))  # set the segment boundaries for the last segment

    assert len(grouped_mags) == len(grouped_angles)
    assert len(grouped_mags) == len(segment_timestamps)

    return grouped_mags, grouped_angles, segment_timestamps


def find_max_magnitude(grouped_mags, magnitude_bins, grouped_angles, segment_timestamps):

    # go through the bins in reverse
    reversed_magnitude_bins = list(reversed(magnitude_bins))
    nmm = False     # is set to "True" if a new max magnitude is found
    new_max_magnitude = 0
    for i, b in enumerate(reversed_magnitude_bins):
        if i == 0:
            for j, m in enumerate(grouped_mags):
                # if a segment that falls into the bin >=b is longer than the step_size stop the search
                # and take the magnitude of this segment as the new max magnitude for scaling
                if m >= b and segment_timestamps[j][1]-segment_timestamps[j][0] > step_size:
                    new_max_magnitude = m
                    nmm = True
                    break
        else:
            for j, m in enumerate(grouped_mags):
                # check if m is higher/equal than the current bin and lower than the previous
                if b <= m < reversed_magnitude_bins[i-1] and segment_timestamps[j][1]-segment_timestamps[j][0] > step_size:
                    new_max_magnitude = m
                    nmm = True
                    break
        if nmm:
            break

    # scale the magnitudes using the new found "maximum" magnitude
    # if no magnitude was found (no segment is longer than the step_size), take the maximum magnitude of the movie as the factor
    if new_max_magnitude == 0:
        new_max_magnitude = np.max(grouped_mags)
        scaled_mags = grouped_mags / new_max_magnitude
        print('no new maximum magnitude could be found, ')
    else:
        print('a new max magnitude was found')
        scaled_mags = grouped_mags / new_max_magnitude

    # clip values greater than 1 to 1
    clipped_mags = np.clip(scaled_mags, 0, 1)
    digitized_mags = [bin_values(x, BINS) for x in clipped_mags]

    # scale the magnitudes that are corresponding to the angles and digitize them
    for i, al in enumerate(grouped_angles):
        for key_angles in al:
            aux = grouped_angles[i][key_angles][1] / new_max_magnitude
            aux = np.clip(aux, 0, 1)
            aux = bin_values(aux, BINS)
            grouped_angles[i][key_angles][1] = aux

    assert len(digitized_mags) == len(grouped_angles)

    return digitized_mags, grouped_angles


def find_dominant_movement(grouped_angles):
    dominant_angle_per_segment = []
    meta_info = []
    for i, histogram in enumerate(grouped_angles):
        aux_histogram = {k: v[0]*v[1] for k, v in histogram.items()}     # multiply the magnitude and angle count to get a measure of movement for each angle
        hist_mean = np.mean(list(aux_histogram.values()))
        hist_var = np.var(list(aux_histogram.values()))
        aux_histogram = {k: ((v-hist_mean)/hist_var if hist_var != 0 else 0) for k, v in aux_histogram.items()}     # normalize the histogram

        reversed_histogram = {v: k for k, v in aux_histogram.items()}   # reverse the histogramm to allow it to be searched by the value
        dominant = sorted(aux_histogram.values(), reverse=True)[0]   # sort the histogram by magnitude times angle, descending, return the biggest value
        dominant_angle_per_segment.append(reversed_histogram[dominant])   # return the angle corresponding to the value

        # add additional information about the segment, whole histogram, min/max/average-values, variance
        info = [np.min(list(aux_histogram.values())), np.max(list(aux_histogram.values())),
                np.mean(list(aux_histogram.values())), np.var(list(aux_histogram.values()))]
        meta_info.append(info)

    assert len(dominant_angle_per_segment) == len(meta_info)

    return dominant_angle_per_segment, meta_info


def write_mag_to_csv(f_path, grouped_mags, segment_timestamps):
    if not os.path.isdir(os.path.join(f_path, 'optical_flow')):
        os.makedirs(os.path.join(f_path, 'optical_flow'))

    if len(grouped_mags) > 500:
        pieces = int((len(grouped_mags) / 500))

        for p in range(pieces):
            mag_csv_path = os.path.join(f_path, 'optical_flow/mag_{}_optical_flow_{}.csv'.format(p, os.path.split(f_path)[1]))
            aux_mag = grouped_mags[p * 500:(p + 1) * 500]
            aux_ts = segment_timestamps[p * 500:(p + 1) * 500]
            with open(mag_csv_path, 'w', newline='') as f:
                for i, mag in enumerate(aux_mag):
                    line = str(aux_ts[i][0]) + ' ' + str(aux_ts[i][1]) + ' ' + str(mag)
                    f.write(line)
                    f.write('\n')

    else:
        mag_csv_path = os.path.join(f_path, 'optical_flow/mag_optical_flow_{}.csv'.format(os.path.split(f_path)[1]))

        with open(mag_csv_path, 'w', newline='') as f:
            for i, mag in enumerate(grouped_mags):
                line = str(segment_timestamps[i][0]) + ' ' + str(segment_timestamps[i][1]) + ' ' + str(mag)
                f.write(line)
                f.write('\n')


def write_angle_to_csv(f_path, dominant_angle_per_segment, angle_meta_info, segment_timestamps):
    if not os.path.isdir(os.path.join(f_path, 'optical_flow')):
        os.makedirs(os.path.join(f_path, 'optical_flow'))

    if len(dominant_angle_per_segment) > 500:
        pieces = int((len(dominant_angle_per_segment) / 500))

        for p in range(pieces):
            aux_angles = dominant_angle_per_segment[p * 500:(p + 1) * 500]
            aux_info = angle_meta_info[p * 500:(p + 1) * 500]
            aux_ts = segment_timestamps[p * 500:(p + 1) * 500]

            angle_csv_path = os.path.join(f_path, 'optical_flow/angle_{}_optical_flow_{}.csv'.format(p, os.path.split(f_path)[1]))

            with open(angle_csv_path, 'w', newline='') as f:
                for i, angle in enumerate(aux_angles):
                    ami = str(aux_info[i]).replace(' ', '')
                    line = str(aux_ts[i][0]) + ' ' + str(aux_ts[i][1]) + ' ' + str(angle) + ' ' + ami
                    f.write(line)
                    f.write('\n')

    else:
        angle_csv_path = os.path.join(f_path, 'optical_flow/angle_optical_flow_{}.csv'.format(os.path.split(f_path)[1]))

        with open(angle_csv_path, 'w', newline='') as f:
            for i, angle in enumerate(dominant_angle_per_segment):
                ami = str(angle_meta_info[i]).replace(' ', '')
                line = str(segment_timestamps[i][0]) + ' ' + str(segment_timestamps[i][1]) + ' ' + str(angle) + ' ' + ami
                f.write(line)
                f.write('\n')


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
                summed_mags, angles_histogram_list, timestamps = get_optical_flow(v_path, frame_width)
                # print('summed mags, angles, timestamps: ', np.shape(summed_mags), np.shape(angles_histogram_list), np.shape(timestamps))

                print('aggregate segments')
                aggregated_segments, magnitude_bins, aggregated_timestamps = aggregate_segments(summed_mags, timestamps)
                # print('aggregated_segments, aggregated_timestamps', np.shape(aggregated_segments), np.shape(aggregated_timestamps))

                print('group angles and manitudes into segments')
                grouped_mags, grouped_angles, segment_timestamps = group_angles_and_magnitudes(aggregated_segments, angles_histogram_list, aggregated_timestamps)
                # print('mags, angles, timestamps: ', np.shape(grouped_mags), np.shape(grouped_angles), np.shape(segment_timestamps))

                print('find max magnitude')
                digitized_mags, new_grouped_angles = find_max_magnitude(grouped_mags, magnitude_bins, grouped_angles, segment_timestamps)

                print('group angles and magnitudes into segments with scaled mags')
                grouped_mags, grouped_angles, segment_timestamps = group_angles_and_magnitudes(digitized_mags, new_grouped_angles, segment_timestamps)

                print('find dominant movements')
                dominant_angle_per_segment, angle_meta_info = find_dominant_movement(grouped_angles)

                print('write results to csv')
                write_mag_to_csv(f_path, grouped_mags, segment_timestamps)
                write_angle_to_csv(f_path, dominant_angle_per_segment, angle_meta_info, segment_timestamps)

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
    parser.add_argument("--step_size", type=int, default=1000, help="")
    parser.add_argument("--window_size", type=int, default=1000, help="")
    args = parser.parse_args()

    step_size = args.step_size
    window_size = args.window_size

    main(args.videos_dir, args.features_dir, args.frame_width)
