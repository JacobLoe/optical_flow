import numpy as np
import cv2
import os
import argparse
import glob
from tqdm import tqdm
import shutil
from scipy.spatial.distance import euclidean

from idmapper import  TSVIdMapper

BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]     #
ANGLE_BINS = [0, 45, 90, 135, 180, 225, 270, 315, 360]

EXTRACTOR = "opticalflow"
VERSION = '20200812'      # the version of the script
aggregate = np.mean


# FIXME: class version for extractors
# FIXME: replace prints by logger

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


def get_optical_flow(v_path, frame_width):

    vid = cv2.VideoCapture(v_path)
    if not vid.isOpened():
        raise IOError("Unable to read from video: '{v_path}'".format(v_path=v_path))

    tot_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS) 
    step_size_in_frames = int(fps*step_size/1000)  # convert the step_size and window_size from ms to frames, dependent on the fps of the movie
    window_size_in_frames = int(fps*window_size/1000)
 
    windows = [( max(0,c-int(window_size_in_frames/2.)), min(tot_frames-1,c+int(window_size_in_frames/2.)) ) for c in range(0,tot_frames,step_size_in_frames)]

    mags = list()
    for start, end in windows:
        if not vid.isOpened():
            raise IOError("Unable to read from video: '{v_path}'".format(v_path=v_path))
        
        ret, start_frame = read_frame(vid, start, frame_width)
        if not ret or start_frame is None:
            raise IOError("Unable to read start frame {f} from video: '{v_path}'".format(f=start, v_path=v_path))

        ret, end_frame = read_frame(vid, end, frame_width)
        
        if not ret or end_frame is None:
            raise IOError("Unable to read end frame {f} from video: '{v_path}'".format(f=end, v_path=v_path))

        mag = calculate_optical_flow(start_frame, end_frame)
        mags.append((start,end,mag))

    vid.release()
    cv2.destroyAllWindows()

    agg_mags = list()
    for pos in range(0,tot_frames,step_size_in_frames):
        agg_mag = [mag[2] for mag in mags if pos>=mag[0] and pos<mag[1]]
        if len(agg_mag)>0:
            agg_mags.append((pos, np.mean(agg_mag)))
        else:
            print("WARN: no entry for pos={pos}".format(pos=pos))

    start_ms = int(agg_mags[0][0]/fps*1000)
    end_ms = int(agg_mags[-1][0]/fps*1000)

    return [mag[1] for mag in agg_mags], [start_ms, end_ms]

	
#    # go through the video and extract two frames, one at the current timestamp, the second at timestamp+window_size
#    # save the optical flow at each timestamp
#    while vid.isOpened():
#        # read the frame at the current timestamp, stop the reading if the video is finished
#        ret, curr_frame = read_frame(vid, timestamp_frames - int(window_size_in_frames/2), frame_width)
#        if not ret:
#            break
#        # read the "future" frame, at the end of the window, stop the reading if the video is finished
#        ret, future_frame = read_frame(vid, timestamp_frames + int(window_size_in_frames/2), frame_width)
#        if not ret:
#            break
#        # calculate the optical flow for the current and the future frame, return the summed up magnitudes and a histogram for the angles found between the frames
#        mag = calculate_optical_flow(curr_frame, future_frame)
#
#        # save the two optical flow components
#        summed_mags.append(mag)
#        # move along the video according to the step_size
#        timestamp_frames += step_size_in_frames
#
#    timestamps.append((int(timestamp_frames/vid.get(cv2.CAP_PROP_FPS)*1000)))
#
#    vid.release()
#    cv2.destroyAllWindows()
#
#    return summed_mags, timestamps


def aggregate_segments(summed_mags):
    aggregated_segments = []   # list of aggregated magnitudes

    ratio = window_size / step_size     # determines whether segments are aggregated or not, values <=1 mean no aggregation is happening (no segments are overlapping)
    segments_to_aggregate = int(ratio)     # determines how many segments are aggregated together
    if segments_to_aggregate == 1 and ratio > 1:   # if the ratio is greater than 1 there are segments to be aggregated, but the ratio would be rounded down to 1 in the next step
        segments_to_aggregate = 2                  # and prevent segment aggregation. To prevent this from happening segments_to_aggregate is set to 2

    step = int(segments_to_aggregate/2)

    if ratio <= 1:  # do nothing if the segments are not overlapping
        aggregated_segments = summed_mags
    else:
        for i in range(len(summed_mags)):
            if i < (segments_to_aggregate-1):   # aggregate at timestamps where less segments then segments_to_aggregate where extracted

                overlapping_segments = summed_mags[i:i+i+1]     # take the the number of overlapping segments at the timestamp (less than segments_to_aggregate
                                                                # only "future" timestamps are used here
                aggregated_mags = aggregate(overlapping_segments)      # with the aggregate function, create one value
                aggregated_segments.append(aggregated_mags)
            else:
                overlapping_segments = summed_mags[i-step:i+step]     # get the overlapping segments from summed mags
                aggregated_mags = aggregate(overlapping_segments)      # with the aggregate function, create one value
                aggregated_segments.append(aggregated_mags)

    return aggregated_segments


def scale_magnitudes(mag, top_percentile):
    scaled_mag = mag / np.percentile(mag, top_percentile)
    scaled_mag = np.clip(scaled_mag, a_min=0, a_max=1)*100.
    scaled_mag = list(np.round(scaled_mag, decimals=2))

    return scaled_mag


def write_mag_to_csv(f_path, mag, segment_timestamps):
    #if not os.path.isdir(os.path.join(f_path, 'optical_flow')):
    #    os.makedirs(os.path.join(f_path, 'optical_flow'))

    #mag_csv_path = os.path.join(f_path, 'optical_flow/mag_optical_flow_{}.csv'.format(os.path.split(f_path)[1]))

    with open(f_path, 'w', newline='') as f:
        mag = " ".join(mag)
        line = str(segment_timestamps[0]) + '\t' + str(segment_timestamps[1]) + '\t' + mag
        f.write(line)


def main(videos_root, features_root, videoids, idmapper, frame_width=129, step_size=300, window_size=300):
    print("Computing optical flow for {0} videos".format(len(videoids)))
    done = 0
    while done < len(videoids):
        for videoid in tqdm(videoids):
            try:
                video_rel_path = idmapper.get_filename(videoid)
            except KeyError as err:
                print("No such videoid: '{videoid}'".format(videoid=videoid))
                done += 1

            print(videoid, os.path.basename(video_rel_path))
            
            features_dir = os.path.join(features_root,videoid,EXTRACTOR)

            if not os.path.isdir(features_dir):
               os.makedirs(features_dir)

            #FIXME: extractor as class, "opticalflow" as property, version as property
            features_fname_vid = "{videoid}.opticalflow.csv".format(videoid=videoid)
            #features_fname_vfn = "{video_fname}.opticalflow.csv".format(video_fname=os.path.splitext(os.path.basename(video_rel_path))[0])
            f_path_vid = os.path.join(features_dir, features_fname_vid)
            #f_path_vfn = os.path.join(features_dir, features_fname_vfn)
            done_file_path = os.path.join(features_dir, '.done')

            v_path = os.path.join(videos_root, video_rel_path)

            if not os.path.isfile(done_file_path) or not open(done_file_path, 'r').read() == VERSION:
                print("Optical flow results missing or version did not match")

                print('Compute angles and magnitudes')
                summed_mags, timestamps = get_optical_flow(v_path, frame_width)

                #print('aggregate segments')
                #aggregated_segments = aggregate_segments(summed_mags)
                aggregated_segments = summed_mags

                scaled_segments = scale_magnitudes(aggregated_segments, top_percentile)

                print('Write results to csv')
                write_mag_to_csv(f_path_vid, scaled_segments, timestamps)

                # create a hidden file to signal that the optical flow for a movie is done
                # write the current version of the script in the file
                with open(done_file_path, 'w') as d:
                    d.write(VERSION)
                done += 1  # count the instances of the optical flow done correctly
            else:
                # do nothing if a .done-file exists and the versions in the file and the script match
                done += 1  # count the instances of the optical flow done correctly
                print('optical flow was already done for {video}'.format(video=video_rel_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("videos_dir", help="the directory where the video-files are stored")
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    parser.add_argument("videoids", help="List of video ids. If empty, entire corpus is iterated.", nargs='*')
    parser.add_argument("--frame_width", type=int, default=129, help="set the width at which to which the frames are rescaled, default is 129")
    parser.add_argument("--step_size", type=int, default=300, help="defines at which distances the optical flow is calculated, in milliseconds, default is 300")
    parser.add_argument("--window_size", type=int, default=300,
                        help="defines the range in which images for optical flow calculation are extracted,"
                             " if window_size is equal to step_size two frames are extracted, default is 300")
    parser.add_argument("--top_percentile", type=int, default=5, help="set the percentage of magnitudes that are used to determine the max magnitude,"
                                                                      "")
    args = parser.parse_args()

    step_size = args.step_size
    window_size = args.window_size
    top_percentile = args.top_percentile

    # FIXME: make more generic once workflow is setup
    idmapper = TSVIdMapper('/root/file_mappings.tsv')
    videoids = args.videoids if len(args.videoids)>0 else idmapper.get_ids()
    main(args.videos_dir, args.features_dir, videoids, idmapper, args.frame_width, args.step_size, args.window_size)
