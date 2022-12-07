'''This script visualizes the tracker output of SORT.
Ran the script of the official repo, as of commit da0fe4d20ff3ca1c7bad3f746fa79e3c97421bf2.
The command was `python sort.py --seq_path ../bsc-thesis/tracking/detections/ --phase {model_name} --max_age 3`
We expect:
    tracker output to be in `trck_path`,
    the corresponding frames from the UA-DETRAC sequence in `vid_folder`.
The output will be written to `out_folder` (if already a directory, the script will ask if you want to overwrite).'''
import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw
import string
import numpy as np
from colorhash import ColorHash

def parse_frame_no(filename: str): return int(filename.strip(string.ascii_letters + '0' + '.'))

def draw_tracking(img: Image, tracks: np.ndarray):
    '''tracks is a Nx5 array, with first column being the object ID, then the box in xmin, ymin, w, h format.'''
    draw = ImageDraw.Draw(img)
    for trck in tracks:
        ID, det = trck[0], trck[1:].copy()
        det[2:] += det[:2] # convert to xyxy format
        det = list(det.astype(int)) # needed by draw.rectangle
        color = ColorHash(ID).rgb
        draw.rectangle(det, outline=color, width=3)
    return img

def visualize_tracking(tracking_file_path: str, video_folder: str, out_folder: str):
    trck = pd.read_csv(
        tracking_file_path, header=None, names=['frame', 'ID', 'xmin', 'ymin', 'w', 'h', '_1', '_2', '_3', '_4'])
    for img_file in tqdm(sorted(os.listdir(video_folder))):
        # get tracker hypotheses for the given frame
        frame_no = parse_frame_no(img_file)
        tracks = trck[['ID', 'xmin', 'ymin', 'w', 'h']][trck['frame']==frame_no].to_numpy()
        path = osp.join(video_folder, img_file)
        with Image.open(path) as img:
            draw_tracking(img, tracks)
            img.save(osp.join(out_folder, img_file))

if __name__ == '__main__':
    # choose setup
    model_name = 'yolov5x'
    sequence_name = 'MVI_63544'
    conf_threshold = 0.8
    # paths
    trck_path = f'tracking/sort_output/{model_name}/{sequence_name}_{conf_threshold:.1f}.txt'
    vid_folder = f'data/DETRAC-train-data/Insight-MVT_Annotation_Train/{sequence_name}'
    out_folder = f'tracking/sort_visualization/{model_name}_{sequence_name}_{conf_threshold:.1f}'
    # comment out if unneded:
    if osp.isdir(out_folder):
        ans = input(f'You are going to overwrite contents of {out_folder}. Proceed? [y/n]: ')
        if ans != 'y': exit()
    else:
        os.mkdir(out_folder)
    # do visualize:
    visualize_tracking(tracking_file_path=trck_path, video_folder=vid_folder, out_folder=out_folder)