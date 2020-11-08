import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
from scipy.spatial.distance import euclidean

from idmapper import TSVIdMapper

BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ANGLE_BINS = [0, 45, 90, 135, 180, 225, 270, 315, 360]

EXTRACTOR = "opticalflow"
VERSION = '20200930'      # the version of the script
STANDALONE = True   # manages the creation of .done-files, if set to True no .done-files are created and the script will always overwrite old results

# FIXME: class version for extractors
# FIXME: replace prints by logger


def resize_frame(frame, frame_width):
    resolution_old = np.shape(frame)
    ratio = resolution_old[1] / resolution_old[0]
    frame_height = int(frame_width / ratio)
    resolution_new = (frame_width, frame_height)
    resized_frame = cv2.resize(frame, resolution_new)
    return resized_frame


def read_frame(vid, timestamp, frame_width):
    # hint: use nframes = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    vid.set(cv2.CAP_PROP_POS_FRAMES, timestamp)
    ret, frame = vid.read()  # if ret is false, frame has no content

    if not ret:
        return ret, None 

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


def get_optical_flow(v_path, frame_width, video_name):

    vid = cv2.VideoCapture(v_path)
    if not vid.isOpened():
        raise IOError("Unable to read from video: '{v_path}'".format(v_path=v_path))

    tot_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS) 
    step_size_in_frames = int(fps*step_size/1000)  # convert the step_size and window_size from ms to frames, dependent on the fps of the movie
    window_size_in_frames = int(fps*window_size/1000)
 
    windows = [(max(0, c-int(window_size_in_frames/2.)), min(tot_frames-1, c+int(window_size_in_frames/2.))) for c in range(0, tot_frames, step_size_in_frames)]

    mags = list()
    for start, end in windows:
        if not vid.isOpened():
            raise IOError("Unable to read from video: '{v_path}'".format(v_path=v_path))

        # if either frame is missing the optical flow can not the be computed
        # it is then assumed that the end of the video is reached
        ret, start_frame = read_frame(vid, start, frame_width)
        if not ret or start_frame is None:
            break

        ret, end_frame = read_frame(vid, end, frame_width)
        
        if not ret or end_frame is None:
            break

        mag = calculate_optical_flow(start_frame, end_frame)
        mags.append((start, end, mag))
    # raise an exception if the no magnitudes where computed
    if not mags:
        raise Exception('Unable to extract the optical flow from "{videoname}", no frames where found.'.format(videoname=video_name))
    vid.release()
    cv2.destroyAllWindows()

    agg_mags = list()
    for pos in range(0, tot_frames, step_size_in_frames):
        agg_mag = [mag[2] for mag in mags if pos>=mag[0] and pos<mag[1]]
        if len(agg_mag)>0:
            agg_mags.append((pos, np.mean(agg_mag)))
        else:
            print("WARN: no entry for pos={pos}".format(pos=pos))

    start_ms = int(agg_mags[0][0]/fps*1000)
    end_ms = int(agg_mags[-1][0]/fps*1000)

    return [mag[1] for mag in agg_mags], [start_ms, end_ms]


def scale_magnitudes(mag, top_percentile):
    scaled_mag = mag / np.percentile(mag, top_percentile)
    scaled_mag = np.clip(scaled_mag, a_min=0, a_max=1)*100.
    scaled_mag = list(np.round(scaled_mag, decimals=2))

    return scaled_mag


def write_mag_to_csv(f_path, mag, segment_timestamps):
    with open(f_path, 'w', newline='') as f:
        mag = " ".join([str(m) for m in mag])
        line = str(segment_timestamps[0]) + '\t' + str(segment_timestamps[1]) + '\t' + mag
        f.write(line)


def main(videos_root, features_root, frame_width, videoids, idmapper):
    print("Computing optical flow for {0} videos".format(len(videoids)))
    for videoid in tqdm(videoids):
        try:
            video_rel_path = idmapper.get_filename(videoid)
        except KeyError as err:
            raise Exception("No such videoid: '{videoid}'".format(videoid=videoid))

        video_name = os.path.basename(video_rel_path)[:-4]
        features_dir = os.path.join(features_root, videoid, EXTRACTOR)

        if not os.path.isdir(features_dir):
            os.makedirs(features_dir)

        # FIXME: extractor as class, "opticalflow" as property, version as property
        features_fname_vid = "{videoid}.{extractor}.csv".format(videoid=videoid, extractor=EXTRACTOR)
        # features_fname_vfn = "{video_fname}.{extractor}.csv".format(video_fname=os.path.splitext(os.path.basename(video_rel_path))[0], extractor=EXTRACTOR)
        f_path_csv = os.path.join(features_dir, features_fname_vid)
        # f_path_vfn = os.path.join(features_dir, features_fname_vfn)
        done_file_path = os.path.join(features_dir, '.done')

        v_path = os.path.join(videos_root, video_rel_path)

        # create the version for a run, based on the script version and the used parameters
        done_version = VERSION+'\n'+str(frame_width)+'\n'+str(step_size)+'\n'+str(window_size)+str(top_percentile)

        if not os.path.isfile(done_file_path) or not open(done_file_path, 'r').read() == done_version or force_run:
            print('Optical flow results missing or version did not match, starting extraction for "{video_name}"'.format(video_name=video_name))

            aggregated_segments, timestamps = get_optical_flow(v_path, frame_width, video_name)
            scaled_segments = scale_magnitudes(aggregated_segments, top_percentile)

            write_mag_to_csv(f_path_csv, scaled_segments, timestamps)

            # create a hidden file to signal that the optical flow for a movie is done
            # write the current version of the script in the file
            if STANDALONE:
                with open(done_file_path, 'w') as d:
                    d.write(done_version)
        else:
            # do nothing if a .done-file exists and the versions in the file and the script match
            print('optical flow was already done for "{video}"'.format(video=video_rel_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("videos_dir", help="the directory where the video-files are stored")
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    parser.add_argument('file_mappings', help='path to file mappings .tsv-file')
    parser.add_argument("videoids", help="List of video ids. If empty, entire corpus is iterated.", nargs='*')
    parser.add_argument("--frame_width", type=int, default=129, help="set the width at which to which the frames are rescaled, default is 129")
    parser.add_argument("--step_size", type=int, default=300, help="defines at which distances the optical flow is calculated, in milliseconds, default is 300")
    parser.add_argument("--window_size", type=int, default=300,
                        help="defines the range in which images for optical flow calculation are extracted,"
                             " if window_size is equal to step_size two frames are extracted, default is 300")
    parser.add_argument("--top_percentile", type=int, default=5, help="set the percentage of magnitudes that are used to determine the max magnitude,""")
    parser.add_argument("--force_run", default=False, type=bool, help='sets whether the script runs regardless of the version of .done-files')
    args = parser.parse_args()

    force_run = args.force_run
    step_size = args.step_size
    window_size = args.window_size
    top_percentile = args.top_percentile

    # FIXME: make more generic once workflow is setup
    idmapper = TSVIdMapper(args.file_mappings)
    videoids = args.videoids if len(args.videoids) > 0 else parser.error('no videoids found')

    main(args.videos_dir, args.features_dir, args.frame_width, videoids, idmapper)
