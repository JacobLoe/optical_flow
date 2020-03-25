import numpy as np
import cv2


path = '../videos/movies/Ferguson_Charles_Inside_Job.mp4'
STEP_SIZE = 300
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1]
frame_width = None


def group_movie_magnitudes(digitized_mags, timestamps):
    grouped_mags = []
    shot_timestamps = []
    prev_mag = 0
    start_ms = 0
    end_ms = 0
    # iterate through the magnitudes,
    for i, curr_mag in enumerate(digitized_mags):
        if i == 0:
            grouped_mags.append(curr_mag)    # start the grouped mag list
            prev_mag = curr_mag  # save the first magnitude for later comparison as previous magnitude

        elif prev_mag == curr_mag:  # as long as the previous and current magnitude are the same
            end_ms = timestamps[i]     # set the end of the "current" shot to the timestamp of the current magnitude

        else:   # if the magnitudes deviate add a new entry
            shot_timestamps.append((start_ms, end_ms))  # set the shot boundaries
            start_ms = end_ms   # set the start timestamp of the next shot as the end of the current shot
            end_ms = timestamps[i]  # set the end of the "current" shot to the timestamp of the current magnitude

            grouped_mags.append(curr_mag)  # add the next value to the grouped mag list
            prev_mag = curr_mag  # save the first magnitude for later comparison as previous mag

    shot_timestamps.append((start_ms, end_ms))  # set the shot boundaries for the last shot

    return grouped_mags, shot_timestamps


def get_optical_flow(v_path, frame_width, start_ms, end_ms):

    vid = cv2.VideoCapture(v_path)
    summed_mags = []
    print('fps', vid.get(cv2.CAP_PROP_FPS))
    start_frame = vid.get(cv2.CAP_PROP_FPS)*start_ms/1000
    end_frame = int(vid.get(cv2.CAP_PROP_FPS)*end_ms/1000)
    step_size_in_frames = int(vid.get(cv2.CAP_PROP_FPS)*STEP_SIZE/1000)  # convert the STEP_SIZE from ms to frames, dependent on the fps of the movie
    timestamp_frames = 0 + start_frame
    timestamps = []

    # print('fps', vid.get(cv2.CAP_PROP_FPS), 'step_size', STEP_SIZE, 'step_size_in_frames', step_size_in_frames)
    # iterate through all shots in a movie
    while timestamp_frames < end_frame:
        # Capture frame-by-frame

        vid.set(cv2.CAP_PROP_POS_FRAMES, timestamp_frames)
        ret, curr_frame = vid.read()  # if ret is false, frame has no content

        if not ret:
            break

        if timestamp_frames == 0 + start_frame:

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

            # sum up all the magnitudes between neighboring frames in the shot creating a single value
            # scale the value by the number of pixels in the image to keep the value between 0 and 1
            summed_mags.append(np.sum(mag) / (np.shape(mag)[0] * np.shape(mag)[1]))
            timestamps.append(int(timestamp_frames/vid.get(cv2.CAP_PROP_FPS)*1000))

            prev_frame = curr_frame     # save the current as the new previous frame for the next iteration
        timestamp_frames += step_size_in_frames

    vid.release()
    cv2.destroyAllWindows()
    print('summed_mags', summed_mags)
    # scale to a max-value of 1
    if np.max(summed_mags) != 0:
        scaled_mags = summed_mags/np.max(summed_mags)
    else:
        scaled_mags = summed_mags
    print('np.max', np.max(summed_mags))
    # print('scaled_mags', scaled_mags)

    # digitize the values,
    # FIXME if changed to right=False, indices start at 1 instead of 0
    indices_digitized_mags = np.digitize(scaled_mags, BINS, right=True)    # returns a numpy array with shape of rounded_mags, the indices correspond to the position in BINS
    # print('indices', indices_digitized_mags)
    digitized_mags = [BINS[i] for i in indices_digitized_mags]    # map the magnitudes to the values in BINS
    return digitized_mags, timestamps


shot_1, timestamps = get_optical_flow(path, frame_width, 0, 10500)
print('shot_1', shot_1)
print('timestamps', timestamps)
print('len lists', len(shot_1), len(timestamps))
print('-------------------------')
grouped_shot_0, shot_timestamps_0 = group_movie_magnitudes(shot_1, timestamps)
print('grouped_mags', grouped_shot_0)
print('shot_timestamps', shot_timestamps_0)
print('len lists', len(grouped_shot_0), len(shot_timestamps_0))

