import cv2
import numpy as np
import argparse
import os

STEP_SIZE = 300


def get_optical_flow(v_path, images_path, start_ms, end_ms):

    if not os.path.isdir(images_path):
        os.makedirs(images_path)

    vid = cv2.VideoCapture(v_path)
    start_frame = vid.get(cv2.CAP_PROP_FPS)*start_ms/1000
    end_frame = int(vid.get(cv2.CAP_PROP_FPS)*end_ms/1000)
    step_size_in_frames = int(vid.get(cv2.CAP_PROP_FPS)*STEP_SIZE/1000)  # convert the STEP_SIZE from ms to frames, dependent on the fps of the movie
    timestamp_frames = start_frame

    # iterate through all shots in a movie
    while timestamp_frames < end_frame:
        # Capture frame-by-frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, timestamp_frames)
        ret, curr_frame_BGR = vid.read()  # if ret is false, frame has no content

        if not ret:
            break

        if timestamp_frames == start_frame:

            prev_frame = cv2.cvtColor(curr_frame_BGR, cv2.COLOR_BGR2GRAY)

        else:

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
            hsv[..., 1] = 255
            hsv[..., 0] = ang * 180 / np.pi     # angle is hue, red is 0 deg, green 120, blue 240
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)     # magnitude is value
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            path_flow = os.path.join(images_path, 'flow_' + str(int(timestamp_frames/vid.get(cv2.CAP_PROP_FPS)*1000)) + '.jpeg')
            path_source = os.path.join(images_path, 'source_' + str(int(timestamp_frames/vid.get(cv2.CAP_PROP_FPS)*1000)) + '.jpeg')
            cv2.imwrite(path_flow, rgb)
            cv2.imwrite(path_source, curr_frame_BGR)

            prev_frame = curr_frame     # save the current as the new previous frame for the next iteration
        timestamp_frames += step_size_in_frames


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", help="the directory where the video-files are stored")
    parser.add_argument("images_path", help="the directory where the images are saved")
    parser.add_argument("shot_begin", type=int, help="the begin of a shot in milliseconds")
    parser.add_argument("shot_end", type=int, help="the end of a shot in milliseconds")
    args = parser.parse_args()

    get_optical_flow(args.video_dir, args.images_path, args.shot_begin, args.shot_end)
