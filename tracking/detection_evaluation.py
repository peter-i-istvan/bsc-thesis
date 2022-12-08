import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw
import string
import numpy as np
from colorhash import ColorHash
import scipy
import motmetrics as mm

# TODO: implement detection mistake visualization
# def show_mistakes(img: Image, acc: mm.MOTAccumulator, gt: np.ndarray, tr: pd.DataFrame):
#     draw = ImageDraw.Draw(img)
#     gt_color = (0,255,0)
#     match_color = (0,0,255)
#     fp_color = (255,0,255)
#     fn_color = (0,255,255)
#     for gt_id, tr_id in acc.m.items():
#         gt_xyxy = gt[gt_id].copy()
#         gt_xyxy[2:] += gt_xyxy[:2] # convert to xyxy
#         tr_xyxy = tr[['xmin', 'ymin', 'w', 'h']][tr['ID']==tr_id].to_numpy().astype(int)[0]
#         tr_xyxy[2:] += tr_xyxy[:2]
#         draw.rectangle(list(gt_xyxy), outline=gt_color, width=3)
#         draw.rectangle(list(tr_xyxy), outline=match_color, width=2)
#         img.show('GT')
#         pass

def evaluate_detection(annotation: dict, detection: pd.DataFrame, video_path: str):
    X = annotation['gtInfo']['X'][0][0]
    Y = annotation['gtInfo']['Y'][0][0]
    W = annotation['gtInfo']['W'][0][0]
    H = annotation['gtInfo']['H'][0][0]
    # the above are in foot coordinates, so we'll need to convert:
    xmin = X - W/2
    ymin = Y - H
    frame_nums = annotation['gtInfo']['frameNums'][0][0][0]
    acc = mm.MOTAccumulator(auto_id=True)
    # iterate over all frames
    for i, fn in enumerate(frame_nums):
        current_detections = detection[detection['frame'] == fn]
        gt_labels = np.where(H[i, :] > 0)[0] # Active objects are those with valid height values
        tr_labels = current_detections['ID'].to_numpy()
        gt_xywh = np.transpose(
            np.vstack([
                xmin[i, gt_labels],
                ymin[i, gt_labels],
                W[i, gt_labels],
                H[i, gt_labels]
            ])
        )
        det_xywh = current_detections[['xmin', 'ymin', 'w', 'h']]
        dist = mm.distances.iou_matrix(gt_xywh, det_xywh, max_iou=0.3)
        acc.update(gt_labels, tr_labels, dist)
        # show mistakes
        # img_filename = f'img{fn:>05}.jpg'
        # img_path = osp.join(video_path, img_filename)
        # with Image.open(img_path) as img:
        #     show_mistakes(img, acc, gt_xywh, current_detections)
    false_negatives = list(acc.mot_events['Type']).count('MISS')
    false_positives = list(acc.mot_events['Type']).count('FP')
    true_positives = list(acc.mot_events['Type']).count('MATCH')
    return false_negatives, false_positives, true_positives

def main():
    models_root = 'tracking/detections'
    annotations_root = 'data/DETRAC-Train-Annotations-MAT/'
    videos_root = 'data/DETRAC-train-data/Insight-MVT_Annotation_Train/'
    model_subdirectories = [m for m in os.listdir(models_root) if osp.isdir(osp.join(models_root, m))]
    model_paths = [osp.join(models_root, m) for m in model_subdirectories]
    conf_thresholds = np.linspace(start=0.9, stop=0.0, num=10)
    # initialize stats accumulator - stats[MODEL_NAME][CONF_THRESHOLD] = {'FP': ..., 'FN': ..., 'TP': ...}
    stats = {
        m: {
            c_t: {'FP': 0, 'FN': 0, 'TP': 0}
            for c_t in conf_thresholds
        }
        for m in model_subdirectories
    }
    for ann_filename in (pbar := tqdm(os.listdir(annotations_root))):
        seq_name = ann_filename.split('.')[0] # strip extension
        pbar.set_description(f'Evaluating sequence: {seq_name}')
        ann_path = osp.join(annotations_root, ann_filename)
        video_path = osp.join(videos_root, seq_name)
        annotation = scipy.io.loadmat(ann_path)
        for mp in model_paths:
            model_name = osp.basename(mp)
            for conf_threshold in conf_thresholds:
                detector_output_path = osp.join(mp, f'{seq_name}_{conf_threshold:.1f}', 'det', 'det.txt')
                detection = pd.read_csv(detector_output_path, header=None, names=['frame', 'ID', 'xmin', 'ymin', 'w', 'h', 'confidence', '_2', '_3', '_4'])
                false_negatives, false_positives, true_positives = evaluate_detection(annotation, detection, video_path)
                stats[model_name][conf_threshold]['FN'] = false_negatives
                stats[model_name][conf_threshold]['FP'] = false_positives
                stats[model_name][conf_threshold]['TP'] = true_positives

    p_curve = {
        m: {
        c_t: 0
        for c_t in conf_thresholds
        }
        for m in model_subdirectories
    }
    r_curve = {
        m: {
        c_t: 0
        for c_t in conf_thresholds
        }
        for m in model_subdirectories
    }        
    for model_name in model_subdirectories:
        for conf_threshold in conf_thresholds:
            true_positives = stats[model_name][conf_threshold]['TP']
            false_negatives = stats[model_name][conf_threshold]['FN']
            false_positives = stats[model_name][conf_threshold]['FP']
            prec = true_positives / (false_positives + true_positives)
            rec = true_positives / (false_negatives + true_positives)
            p_curve[model_name][conf_threshold] = prec
            r_curve[model_name][conf_threshold] = rec
    r_curve_df = pd.DataFrame(r_curve)
    p_curve_df = pd.DataFrame(p_curve)
    r_curve_df.to_csv('r_curve.csv')
    p_curve_df.to_csv('p_curve.csv')


if __name__ == '__main__':
    main()
