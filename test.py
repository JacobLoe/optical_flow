from dvt.core import DataExtraction, FrameInput
from dvt.annotate.opticalflow import OpticalFlowAnnotator
from dvt.annotate.hofm import HOFMAnnotator
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("videos_dir", help="the directory where the video-files are stored")
args = parser.parse_args()

print(args.videos_dir)

cached_vid = '/tmp/output.mp4'

vid = cv2.VideoCapture('Her_bluray_Szene_11_25fps.mp4')

if not os.path.isfile(cached_vid):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cached_vid, fourcc, 20.0, (128, 128))

    ret, frame = vid.read()
    while ret:
        ret, frame = vid.read()
        try:
            frame = cv2.resize(frame, (128, 128))
        except:
            pass
        out.write(frame)
    out.release()
    vid.release()
    cv2.destroyAllWindows()

dextra = DataExtraction(
  vinput=FrameInput(input_path=cached_vid)
)

dextra.run_annotators([OpticalFlowAnnotator()])
dextra.get_data()
