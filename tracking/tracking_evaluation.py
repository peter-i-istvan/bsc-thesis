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

# TODO: implement detecttracking mistake visualization
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

def evaluate_tracking(annotation: dict, tracking: pd.DataFrame, video_path: str, name: str):
    X = annotation['gtInfo']['X'][0][0]
    Y = annotation['gtInfo']['Y'][0][0]
    W = annotation['gtInfo']['W'][0][0]
    H = annotation['gtInfo']['H'][0][0]
    # the above are in foot coordinates, so we'll need to convert:
    xmin = X - W/2
    ymin = Y - H
    frame_nums = annotation['gtInfo']['frameNums'][0][0][0]
    acc = mm.MOTAccumulator(auto_id=True)
    for i, fn in enumerate(frame_nums):
        current_tracks = tracking[tracking['frame'] == fn]
        gt_labels = np.where(H[i, :] > 0)[0] # Active objects are those with valid height values
        tr_labels = current_tracks['ID'].to_numpy()
        gt_xywh = np.transpose(
            np.vstack([
                xmin[i, gt_labels],
                ymin[i, gt_labels],
                W[i, gt_labels],
                H[i, gt_labels]
            ])
        )
        tr_xywh = current_tracks[['xmin', 'ymin', 'w', 'h']]
        dist = mm.distances.iou_matrix(gt_xywh, tr_xywh, max_iou=0.3)
        acc.update(gt_labels, tr_labels, dist)
        # show mistakes
        # img_filename = f'img{fn:>05}.jpg'
        # img_path = osp.join(video_path, img_filename)
        # with Image.open(img_path) as img:
        #     show_mistakes(img, acc, gt_xywh, current_tracks)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'num_switches', 'idfp', 'idfn', 'idtp', 'mostly_tracked', 'mostly_lost', 'num_unique_objects'], name=name)
    return summary

def main():

    models_root = 'tracking/sort_output'
    annotations_root = 'data/DETRAC-Train-Annotations-MAT/'
    videos_root = 'data/DETRAC-train-data/Insight-MVT_Annotation_Train/'
    out_path = 'tracking/tracking_evaluations'
    model_subdirectories = [m for m in os.listdir(models_root)]
    model_paths = [osp.join(models_root, m) for m in model_subdirectories]
    conf_thresholds = np.linspace(start=0.9, stop=0.0, num=10)
    # aggregated stats[model][conf_threshold] = list(DataFrame)
    stats = {
        m: {
            c_t: []
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
        # ['num_frames', 'mota', 'motp', 'num_switches', 'idfp', 'idfn', 'idtp', 'mostly_tracked', 'mostly_lost', 'num_unique_objects']
        for mp in model_paths:
            model_name = osp.basename(mp)
            for conf_threshold in conf_thresholds:
                tracker_output_path = osp.join(mp, f'{seq_name}_{conf_threshold:.1f}.txt')
                tracking = pd.read_csv(tracker_output_path, header=None, names=['frame', 'ID', 'xmin', 'ymin', 'w', 'h', '_1', '_2', '_3', '_4'])
                summary = evaluate_tracking(annotation, tracking, video_path, seq_name)
                stats[model_name][conf_threshold].append(summary)
        # aggregate sequences
        aggregates = {
            m: []
            for m in model_subdirectories
        }
        for model_name in model_subdirectories:
            for conf_threshold in conf_thresholds:
                df_list = stats[model_name][conf_threshold]
                total = pd.concat(df_list)
                total.to_csv(osp.join(out_path, f'{model_name}_{conf_threshold:.1f}.csv'))
                aggregate = pd.DataFrame(columns=total.columns)
                aggregate['mota'] = [np.average(total['mota'])]
                aggregate['min_mota'] = [np.min(total['mota'])]
                aggregate['max_mota'] = [np.max(total['mota'])]
                aggregate['motp'] = [np.average(total['motp'])]
                aggregate['num_switches'] = [np.sum(total['num_switches'])]
                aggregate['idfp'] = [np.sum(total['idfp'])]
                aggregate['idfn'] = [np.sum(total['idfn'])]
                aggregate['idtp'] = [np.sum(total['idtp'])]
                aggregate['mostly_tracked'] = [np.sum(total['mostly_tracked'])]
                aggregate['mostly_lost'] = [np.sum(total['mostly_lost'])]
                aggregate['num_unique_objects'] = [np.sum(total['num_unique_objects'])]
                aggregate['conf_threshold'] = [conf_threshold]
                aggregate.to_csv(osp.join(out_path, f'{model_name}_{conf_threshold:.1f}_aggregate.csv'))
                aggregates[model_name].append(aggregate)
        # aggregate all conf thresholds:
        for model_name in model_subdirectories:
            model_aggregate = pd.concat(aggregates[model_name], ignore_index=True)
            model_aggregate.to_csv(f'{model_name}_mota.csv')





if __name__ == '__main__':
    main()
