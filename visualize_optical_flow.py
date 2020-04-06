import cv2
import numpy as np


def get_optical_flow(v_path, frame_width, start_ms, end_ms):

    vid = cv2.VideoCapture(v_path)
    print('fps', vid.get(cv2.CAP_PROP_FPS))
    start_frame = vid.get(cv2.CAP_PROP_FPS)*start_ms/1000
    end_frame = int(vid.get(cv2.CAP_PROP_FPS)*end_ms/1000)
    step_size_in_frames = int(vid.get(cv2.CAP_PROP_FPS)*STEP_SIZE/1000)  # convert the STEP_SIZE from ms to frames, dependent on the fps of the movie
    timestamp_frames = 0 + start_frame

    images_optical_flow = []

    # iterate through all shots in a movie
    while timestamp_frames < end_frame:
        # Capture frame-by-frame

        vid.set(cv2.CAP_PROP_POS_FRAMES, timestamp_frames)
        ret, curr_frame_BGR = vid.read()  # if ret is false, frame has no content

        if not ret:
            break

        if timestamp_frames == 0 + start_frame:

            if frame_width:
                resolution_old = np.shape(curr_frame_BGR)
                ratio = resolution_old[1]/resolution_old[0]
                frame_height = int(frame_width/ratio)
                resolution_new = (frame_width, frame_height)
                curr_frame_BGR = cv2.resize(curr_frame, resolution_new)
            curr_frame = cv2.cvtColor(curr_frame_BGR, cv2.COLOR_BGR2GRAY)
            prev_frame = curr_frame

        else:

            if frame_width:
                resolution_old = np.shape(curr_frame_BGR)
                ratio = resolution_old[1]/resolution_old[0]
                frame_height = int(frame_width/ratio)
                resolution_new = (frame_width, frame_height)
                curr_frame_BGR = cv2.resize(curr_frame, resolution_new)
            curr_frame = cv2.cvtColor(curr_frame_BGR, cv2.COLOR_BGR2GRAY)

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

            # create images to visualize the optical flow
            hsv = np.zeros_like(curr_frame_BGR)
            print(np.shape(hsv))
            print(np.shape(ang))
            hsv[..., 1] = 255
            hsv[..., 0] = ang * 180 / np.pi
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            path = 'test_images/' + str(timestamp_frames) + '.jpeg'
            cv2.imwrite(path, rgb)

            prev_frame = curr_frame     # save the current as the new previous frame for the next iteration
        timestamp_frames += step_size_in_frames


if __name__ == "__main__":

    path = '../videos/movies/Ferguson_Charles_Inside_Job.mp4'
    STEP_SIZE = 300
    BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1]
    frame_width = None

    get_optical_flow(path, frame_width, 0, 10500)
