import os
import os.path as osp
import torch
from torchvision.transforms import ToTensor
from collections import namedtuple
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput
from typing import Union

# relevant classes from the COCO dataset that we are looking for
COCO_CLASSES = {'car': 3, 'bus': 6, 'truck': 8}

# when kind is 'yolo', the model is loaded from torch hub based on repo and name
# when kind is 'detr', the model is loaded from huggingface hub based on repo, name is used when printing statistics.
ModelDesc = namedtuple('ModelDesc', 'kind repo name')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def draw_boxes(img: Image, xyxy: torch.tensor):
    '''Draws the boxes from xyxy to a copy of img, then returns it.'''
    tmp = img.copy()
    draw = ImageDraw.Draw(tmp)
    for det in xyxy:
        draw.rectangle(det)
    return tmp

def write_detection(out_file_path: str, frame_idx: int, valid_detections: np.ndarray):
    valid_detections[:, 2:4] -= valid_detections[:, :2] # x_top_left, y_top_left, w, h
    nb_det = valid_detections.shape[0]
    df = pd.DataFrame({
        'frame_index': [frame_idx] * nb_det,
        'trajectory_id': [-1] * nb_det,
        'xmin': valid_detections[:, 0],
        'ymin': valid_detections[:, 1],
        'w': valid_detections[:, 2],
        'h': valid_detections[:, 3],
        'confidence': 100,
        'N/A_1': [-1] * nb_det,
        'N/A_2': [-1] * nb_det,
        'N/A_3': [-1] * nb_det
    })
    df.to_csv(out_file_path, mode='a', header=False, index=False)

class DETRModel:
    '''Wrapper for the Huggingface transformer version of the DETR model.'''
    def __init__(self, md: ModelDesc):
        self.feature_extractor = DetrFeatureExtractor.from_pretrained(md.repo)
        self.model = DetrForObjectDetection.from_pretrained(md.repo)
        self.model.to(device)

    def post_process(self, results: DetrObjectDetectionOutput, conf_threshold: float, target_size: tuple) -> np.ndarray:
        '''Applies confidence thresholding and returns a numpy array with scaled box coordinates (xmin, ymin, xmax, ymax).'''
        # PIL.Image.size is wh, while the feature extractor expects hw
        target_size = target_size[::-1]
        results = self.feature_extractor.post_process_object_detection(
            results, threshold=conf_threshold, target_sizes=[target_size]
        )[0] # our 'batch' is always of size 1
        keep = (results['labels'] == 3) # keep only relevant results
        valid_boxes = results['boxes'][keep]
        return valid_boxes.detach().numpy()

    
    def __call__(self, img: Image, measure_time: bool = False) -> Union[tuple[DetrObjectDetectionOutput, float], DetrObjectDetectionOutput]:
        '''Returns a DetrObjectDetectionOutput. All its internal tensors are on the CPU when returning.
        For its structure, see: https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/detr#transformers.DetrForObjectDetection.forward
        Also returns the elapsed time of inference, if measure_time is True.
        The time of the inference on GPU is measured with the appropriate torch.cuda tools, and does not include preprocessing time, or copy time to and from the GPU memory.'''
        inputs = self.feature_extractor(images=img, return_tensors='pt')
        # send data on gpu:
        for k in inputs.keys():
            inputs[k] = inputs[k].to(device)
        # https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
        if measure_time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        output = self.model(**inputs)
        if measure_time:
            end.record()
            torch.cuda.synchronize()
            dt = start.elapsed_time(end)
        output.logits = output.logits.cpu()
        output.pred_boxes = output.pred_boxes.cpu()
        if measure_time:
            return output, dt
        else:
            return output

class YOLOModel:
    '''Wrapper for the YOLOv5 model from Torch Hub.'''
    def __init__(self, md: ModelDesc):
        self.model = torch.hub.load(md.repo, md.name)
        self.to_tensor = ToTensor()

    def post_process(self, results: DetrObjectDetectionOutput, conf_threshold: float, _: tuple) -> np.ndarray:
        '''Applies confidence thresholding and returns a numpy array with scaled box coordinates (xmin, ymin, xmax, ymax).'''
        return results[results[:,4]>conf_threshold][:,:4].numpy()

    def __call__(self, img: Image, measure_time: bool = False) -> Union[tuple[torch.tensor, float], torch.tensor]:
        '''Returns a Nx6 torch tensor of detections: (xmin, ymin, xmax, ymax, conf, class). These are returned in pixel coordinates.
        Also returns the elapsed time of inference, if measure_time is True.
        The time of the inference on GPU is measured with the appropriate torch.cuda tools, and does not include copy time to and from the GPU memory.
        Confidence thresholding is not yet applied.'''
        # img = self.to_tensor(img).to(device)
        if measure_time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        result = self.model([img])
        if measure_time:
            end.record()
            torch.cuda.synchronize()
            dt = start.elapsed_time(end)
        if measure_time:
            return result.xyxy[0].cpu().detach(), dt
        else:
            return result.xyxy[0].cpu().detach()
        

def main():
    # set detection targets
    mds = [
        ModelDesc(kind='yolo', repo='ultralytics/yolov5', name='yolov5n'),
        ModelDesc(kind='yolo', repo='ultralytics/yolov5', name='yolov5s'),
        ModelDesc(kind='yolo', repo='ultralytics/yolov5', name='yolov5m'),
        ModelDesc(kind='yolo', repo='ultralytics/yolov5', name='yolov5l'),
        ModelDesc(kind='yolo', repo='ultralytics/yolov5', name='yolov5x'),
        ModelDesc(kind='detr', repo='facebook/detr-resnet-50', name='detr-resnet-50'),
        ModelDesc(kind='detr', repo='facebook/detr-resnet-101', name='detr-resnet-101')
    ]
    sequences_root = 'data/DETRAC-train-data/Insight-MVT_Annotation_Train/'
    conf_thresholds = np.linspace(start=0.9, stop=0.0, num=10)
    save_predictions_on_images = False
    # set output folder:
    out_root = 'tracking/detections/'
    image_inf_time_ms = {} # gather inference times here
    avg_image_inf_time_ms = {} # average and std of inference times
    # run detections
    for md in mds:
        # for every model
        model_path = os.path.join(out_root, md.name)
        os.mkdir(model_path)
        model = YOLOModel(md) if md.kind == 'yolo' else DETRModel(md)
        image_inf_time_ms[md.name] = []
        avg_image_inf_time_ms[md.name] = {'avg': 0.0, 'std': 0.0}
        print(f'Evaluating model {md.repo}/{md.name}')
        for sequence in (pbar := tqdm(os.listdir(sequences_root))):
            # for every sequence
            pbar.set_description(f'Detection: {sequence}')
            seq_path = osp.join(sequences_root, sequence)
            # out_path = osp.join(out_root, sequence)
            # os.mkdir(out_path)
            for idx, filename in enumerate(sorted(os.listdir(seq_path)), 1):
                # for every image
                img_path = osp.join(seq_path, filename)
                with Image.open(img_path) as img:
                    output, dt = model(img, measure_time=True)
                    image_inf_time_ms[md.name].append(dt)
                    for conf_threshold in conf_thresholds:
                        valid_preds = model.post_process(output, conf_threshold, img.size)
                        if save_predictions_on_images:
                            drawn_image = draw_boxes(img, valid_preds)
                            img_out_path = osp.join(out_path, filename)
                            drawn_image.save(img_out_path)
                        # creating SORT input folder structure: SEQUENCE FOLDER
                        out_path = osp.join(model_path, f'{sequence}_{conf_threshold:.1f}')
                        if not osp.isdir(out_path): os.mkdir(out_path)
                        det_path = osp.join(out_path, 'det')
                        if not osp.isdir(det_path): os.mkdir(det_path)
                        det_file_path = osp.join(det_path, 'det.txt')
                        write_detection(det_file_path, idx, valid_preds)
        avg_image_inf_time_ms[md.name]['avg'] = np.average(image_inf_time_ms[md.name])
        avg_image_inf_time_ms[md.name]['std'] = np.std(image_inf_time_ms[md.name])
        print(f'Inference time: avg={avg_image_inf_time_ms[md.name]["avg"]:.2f} ms; std={avg_image_inf_time_ms[md.name]["std"]:.2f} ms')
    inf_all_df_path = osp.join(out_root, 'inf_times_ms.csv')
    inf_stat_df_path = osp.join(out_root, 'inf_stat_ms.csv')
    pd.DataFrame(image_inf_time_ms).to_csv(inf_all_df_path)
    pd.DataFrame(avg_image_inf_time_ms).to_csv(inf_stat_df_path)


if __name__ == '__main__':
    main()