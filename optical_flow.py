import numpy as np
import cv2
import os
import argparse
import glob
from tqdm import tqdm
import shutil

STEP_SIZE = 300     # the steps in ms that are taken in a shot
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1]     #
ANGLE_BINS = [0, 45, 90, 135, 180, 225, 270, 315, 360]


def get_optical_flow(v_path, frame_width):

    vid = cv2.VideoCapture(v_path)
    summed_mags = []
    timestamps = []
    angles_histogram_list = []
    timestamp_frames = 0
    step_size_in_frames = int(vid.get(cv2.CAP_PROP_FPS)*STEP_SIZE/1000)  # convert the STEP_SIZE from ms to frames, dependent on the fps of the movie
    # iterate through all shots in a movie
    while vid.isOpened():
        # Capture frame-by-frame

        vid.set(cv2.CAP_PROP_POS_FRAMES, timestamp_frames)
        ret, curr_frame = vid.read()  # if ret is false, frame has no content

        if not ret:
            break

        if timestamp_frames == 0:

            if frame_width:
                resolution_old = np.shape(curr_frame)
                ratio = resolution_old[1]/resolution_old[0]
                frame_height = int(frame_width/ratio)
                resolution_new = (frame_width, frame_height)
                curr_frame = cv2.resize(curr_frame, resolution_new)
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            prev_frame = curr_frame
        else:

            if frame_width:
                resolution_old = np.shape(curr_frame)
                ratio = resolution_old[1]/resolution_old[0]
                frame_height = int(frame_width/ratio)
                resolution_new = (frame_width, frame_height)
                curr_frame = cv2.resize(curr_frame, resolution_new)
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # create the optical flow for two neighbouring frames
            flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame,
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

            # digitize the angles found in
            ang = ang * 180 / np.pi
            angles_newshape = np.shape(ang)[0]*np.shape(ang)[1]
            ang = np.reshape(ang, newshape=angles_newshape)
            indices_digitized_angles = np.digitize(ang, ANGLE_BINS, right=True)
            digitized_angles = [ANGLE_BINS[i] for i in indices_digitized_angles]

            # create an empty histogram
            angles_histogram = {b:[0, 0] for b in ANGLE_BINS}
            # flatten the magnitudes to the same shape as the digitized_angles
            flattened_mag = np.reshape(mag, newshape=np.shape(mag)[0]*np.shape(mag)[1])
            for i, angle in enumerate(digitized_angles):
                for b in ANGLE_BINS:
                    if b == angle:
                        angles_histogram[b][0] += 1
                        angles_histogram[b][1] += flattened_mag[i]

            angles_histogram_list.append(angles_histogram)

            # sum up all the magnitudes between neighboring frames in the shot creating a single value
            summed_mags.append(np.sum(mag))
            timestamps.append(int(timestamp_frames/vid.get(cv2.CAP_PROP_FPS)*1000))

            prev_frame = curr_frame     # save the current frame as the new previous frame for the next iteration
        timestamp_frames += step_size_in_frames

    vid.release()
    cv2.destroyAllWindows()

    # aux = list(summed_mags)
    # for i in range(3):
    #     max_mag = np.max(aux)
    #     aux.remove(max_mag)

    # scale to a max-value of 1, if the max value is 0 scaling is skipped to avoid nans
    max_summed_mags = np.max(summed_mags)
    if max_summed_mags != 0:
        scaled_mags = summed_mags/max_summed_mags

        # scale the magnitudes that are corresponding to the angles
        for i, al in enumerate(angles_histogram_list):
            for key_angles in al:
                angles_histogram_list[i][key_angles][1] / max_summed_mags
    else:
        scaled_mags = summed_mags

    # FIXME clip values greater 1

    # FIXME if changed to right=False, indices start at 1 instead of 0
    indices_digitized_mags = np.digitize(scaled_mags, BINS, right=True)    # returns a numpy array with shape of rounded_mags, the indices correspond to the position in BINS
    digitized_mags = [BINS[i] for i in indices_digitized_mags]    # map the magnitudes to the values in BINS

    return digitized_mags, angles_histogram_list, timestamps


def group_angles_and_magnitudes(digitized_mags, angles_histogram_list, timestamps):
    print(len(digitized_mags))
    grouped_mags = []
    grouped_angles = []
    shot_timestamps = []
    prev_mag = 0
    start_ms = 0
    end_ms = 0
    j = 0
    # iterate through the magnitudes,
    for i, curr_mag in enumerate(digitized_mags):
        if i == 0:
            grouped_mags.append(curr_mag)    # start the grouped mag list
            prev_mag = curr_mag  # save the first magnitude for later comparison as previous magnitude
            grouped_angles.append(angles_histogram_list[i])   # save the first the first histogram

        elif prev_mag == curr_mag:  # as long as the previous and current magnitude are the same
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

    # aa = [end-begin-STEP_SIZE for begin, end in shot_timestamps]
    # print(shot_timestamps)
    # print('_----------------')
    # print(aa)
    # print(len(shot_timestamps))
    # print(grouped_mags)
    # print(dasfaf)

    print(len(grouped_mags))
    return grouped_mags, grouped_angles, shot_timestamps


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

            if not os.path.isdir(os.path.join(f_path, 'optical_flow')) and not os.path.isfile(os.path.join(f_path, 'optical_flow/.done')):
                print('optical flow is calculated for {}'.format(os.path.split(v_path)[1]))
                os.makedirs(os.path.join(f_path, 'optical_flow'))

                print('get angles and magnitudes')
                digitized_mags, angles_histogram_list, timestamps = get_optical_flow(v_path, frame_width)

                print('group angles and manitudes into shots')
                grouped_mags, grouped_angles, shot_timestamps = group_angles_and_magnitudes(digitized_mags, angles_histogram_list, timestamps)

                print('find dominant movements')
                dominant_angle_per_shot, angle_meta_info = find_dominant_movement(grouped_angles)

                print('write results to csv')
                write_angle_to_csv(f_path, dominant_angle_per_shot, angle_meta_info, shot_timestamps)
                write_mag_to_csv(f_path, grouped_mags, shot_timestamps)

                print(adadasd)
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
    args = parser.parse_args()

    main(args.videos_dir, args.features_dir, args.frame_width)

