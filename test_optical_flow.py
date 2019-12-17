from dvt.core import DataExtraction, FrameInput
# from dvt.annotate.opticalflow import OpticalFlowAnnotator
from dvt.annotate.hofm import HOFMAnnotator
import cv2
import os
import argparse
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm
import pickle

def read_shotdetect_xml(path):
    tree = ET.parse(path)
    root = tree.getroot().findall('content')
    timestamps = []
    for child in root[0].iter():
        if child.tag == 'shot':
            items = child.items()
            timestamps.append((int(items[4][1]), int(items[4][1])+int(items[2][1])-1))  # ms
    return timestamps  # in ms

def get_optical_flow(v_path, f_path, video_resolution):
    shot_timestamps = read_shotdetect_xml(os.path.join(f_path, 'shot_detection/result.xml'))

    vid = cv2.VideoCapture(v_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    of_annotations = []
    for start_frame, end_frame in tqdm(shot_timestamps):
        cached_vid = '/tmp/'+str(start_frame)+'.mp4'

        # save the shot as individual videos in/tmp
        # the resolution needs to be evenly divisible by the blocks parameter of the HOFMAnnotator
        out_video = cv2.VideoWriter(cached_vid, fourcc, 20.0, video_resolution)

        for i in range(int(end_frame / 1000 - start_frame / 1000) + 1):
            if not (start_frame / 1000 + i) == (int(end_frame / 1000 - start_frame / 1000) + 1):
                vid.set(cv2.CAP_PROP_POS_MSEC, start_frame + i * 1000)
                ret, frame = vid.read()
                frame = cv2.resize(frame, video_resolution)
                out_video.write(frame)

        out_video.release()
        # run the dvt optical flow annotator
        dextra = DataExtraction(
            vinput=FrameInput(input_path=cached_vid)
        )

        # dextra.run_annotators([OpticalFlowAnnotator()])
        dextra.run_annotators([HOFMAnnotator(name='OpticalFlowHistogram', blocks=3)])

        of_annotations.append(dextra.get_data())
        optical_flow_path = os.path.join(f_path, 'optical_flow', str(start_frame))
        with open(optical_flow_path, 'wb') as handle:
            pickle.dump(dextra.get_data(), handle)

    vid.release()
    cv2.destroyAllWindows()
    # return of_annotations



def main(videos_path,features_path, video_resolution):
    list_videos_path = glob.glob(os.path.join(videos_path, '**/*.mp4'), recursive=True)  # get the list of videos in videos_dir

    cp = os.path.commonprefix(list_videos_path)  # get the common dir between paths found with glob

    list_features_path = [os.path.join(
                         os.path.join(features_path,
                         os.path.relpath(p, cp))[:-4])  # add a new dir 'VIDEO_FILE_NAME/shot_detection' to the path
                         for p in list_videos_path]  # create a list of paths where all the data (shot-detection,frames,features) are saved to

    for v_path, f_path in zip(list_videos_path, list_features_path):
        get_optical_flow(v_path, f_path, video_resolution)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("videos_dir", help="the directory where the video-files are stored")
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    args = parser.parse_args()

    video_resolution = (129, 129)
    main(args.videos_dir, args.features_dir, video_resolution)

