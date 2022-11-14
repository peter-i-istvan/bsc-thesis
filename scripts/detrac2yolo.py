import argparse
from configparser import ConfigParser
import json
from dataclasses import dataclass
import os
import shutil
import scipy.io
from tqdm import tqdm
import numpy as np

# these values are from the official website: https://detrac-db.rit.albany.edu/
IMAGE_WIDTH = 960.0
IMAGE_HEIGHT = 540.0

@dataclass
class Split:
    train:  list[str]
    val:    list[str]

@dataclass(kw_only=True)
class Folders:
    images_root:            str
    labels_root:            str
    output_dataset_root:    str
    train_dataset_root:     str
    val_dataset_root:       str
    train_images_root:      str
    train_labels_root:      str
    val_images_root:        str
    val_labels_root:        str

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Transform the UA-DETRAC dataset into a YOLOv7 dataset')
    parser.add_argument('--output_dataset', '-o', type=str, default='detrac_yolo', help='Root folder of the output dataset.')
    parser.add_argument('--images', type=str, default='data/DETRAC-train-data/Insight-MVT_Annotation_Train/', help='Training images root folder')
    parser.add_argument('--labels', type=str, default='data/DETRAC-Train-Annotations-MAT', help='Training annotations root folder')
    return parser.parse_args()

def read_split(conf: ConfigParser) -> Split:
    train = json.loads(conf['split']['train'])
    val = json.loads(conf['split']['val'])
    return Split(train, val)

def mkdir_warn(path: str):
    if os.path.isdir(path):
        ans = input(f'Do you wish to overwrite contents of "{path}"? (y/n): ')
        if not ans == 'y':
            print('Exiting script...')
            exit(1)
        else:
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)

def make_foldernames(images_root: str, labels_root: str, output_dataset_root: str) -> Folders:
    train_dataset_root = os.path.join(output_dataset_root, 'train')
    val_dataset_root = os.path.join(output_dataset_root, 'val')
    return Folders(
        images_root=images_root,
        labels_root=labels_root,
        output_dataset_root=output_dataset_root,
        train_dataset_root=train_dataset_root,
        val_dataset_root=val_dataset_root,
        train_images_root=os.path.join(train_dataset_root, 'images'),
        train_labels_root=os.path.join(train_dataset_root, 'labels'),
        val_images_root=os.path.join(val_dataset_root, 'images'),
        val_labels_root=os.path.join(val_dataset_root, 'labels')
    )

def make_folders(folders: Folders):
    mkdir_warn(folders.output_dataset_root) # delete old one if exists
    os.mkdir(folders.train_dataset_root)
    os.mkdir(folders.train_images_root)
    os.mkdir(folders.train_labels_root)
    os.mkdir(folders.val_dataset_root)
    os.mkdir(folders.val_images_root)
    os.mkdir(folders.val_labels_root)

def get_sequence_annotation(labels_root: str, sequence: str) -> tuple[np.ndarray]:
    '''Reads MAT annotation files for the given sequence. Performs existence and consistency checks.
    Returns a tuple of five numpy arrays: X, Y, W, H, frameNums'''
    labels_file_path = os.path.join(labels_root, f'{sequence}.mat')
    assert os.path.isfile(labels_file_path), f'{labels_file_path} not found'
    annotation = scipy.io.loadmat(labels_file_path)
    X = annotation['gtInfo']['X'][0][0]
    Y = annotation['gtInfo']['Y'][0][0]
    W = annotation['gtInfo']['W'][0][0]
    H = annotation['gtInfo']['H'][0][0]
    frameNums = annotation['gtInfo']['frameNums'][0][0]
    # frame numbers in sequence match
    assert X.shape[0] == Y.shape[0] == W.shape[0] == H.shape[0] == frameNums.shape[1]
    # entity numbers match
    assert X.shape[1] == Y.shape[1] == W.shape[1] == H.shape[1]
    return X, Y, W, H, frameNums

def convert2center(labels: np.ndarray) -> np.ndarray:
    '''Converts rows of [classid, xmin, ymin, w, h] to [classid, xcenter, ycenter, w, h]'''
    new_labels = labels.copy() # for safety's sake
    new_labels[:, 1:3] = labels[:, 1:3] + labels[:, 3:5]/2.
    return new_labels

def matrix2string(m: np.ndarray) -> str:
    '''Converts numpy matrix to custom string format'''
    assert m.ndim == 2
    raw = np.array2string(m)
    raw = raw.replace('[', '').replace(']', '')                 # remove brackets
    stripped = '\n'.join([s.strip() for s in raw.split('\n')])    # strip leading trailing
    return stripped

def process_folders(folders: Folders, split: Split):
    # Go through each subfolder as a sequence of frames, and their annotations
    for subdir in (pbar := tqdm(os.listdir(folders.images_root))):
        if subdir in split.train:
            dest_images_root = folders.train_images_root
            dest_labels_root = folders.train_labels_root
            split_name = 'train'
        elif subdir in split.val:
            dest_images_root = folders.val_images_root
            dest_labels_root = folders.val_labels_root
            split_name = 'val'
        else:
            raise ValueError('Sequence must be either train or validation data')
        pbar.set_description(f'Sequence: {subdir} ({split_name})')
        X, Y, W, H, frameNums = get_sequence_annotation(folders.labels_root, subdir)
        # Make images and labels file under destination
        for i, fn in enumerate(frameNums[0]):
            # we assume 'imgXXXXX.jpg' filename, number always 5 digits long
            image_file_path = os.path.join(folders.images_root, subdir, f'img{fn:0>5}.jpg')
            assert os.path.isfile(image_file_path), f'{image_file_path} not found'
            image_destination_path = os.path.join(dest_images_root, f'{subdir}_{fn}.jpg')
            # Copy the image
            shutil.copy(src=image_file_path, dst=image_destination_path)
            # Copy the labels into a file of space-separated values
            labels_destination_path = os.path.join(dest_labels_root, f'{subdir}_{fn}.txt')
            with open(labels_destination_path, 'w') as file:
                # convert everything to relative coordinates:
                to_stack = [
                    np.zeros((X.shape[1], 1), dtype=float), # classid as col. vector
                    X[i,:].reshape((-1, 1))/IMAGE_WIDTH,    # xmin as col. vector
                    Y[i,:].reshape((-1, 1))/IMAGE_HEIGHT,   # ymin as col. vector
                    W[i,:].reshape((-1, 1))/IMAGE_WIDTH,    # width as col. vector
                    H[i,:].reshape((-1, 1))/IMAGE_HEIGHT    # heigth as col. vector
                ]
                annot = np.hstack(to_stack)
                annot = annot[~np.all(annot == 0, axis=1)]      # remove all 0. rows
                annot = convert2center(annot)
                file_content = matrix2string(annot)
                file.write(file_content)    

def write_yaml(folders: Folders, filename: str = 'custom'):
    file_path = os.path.join(folders.output_dataset_root, f'{filename}.yaml')
    with open(file_path, 'w') as file:
        train_abspath = os.path.abspath(folders.train_dataset_root)
        val_abspath = os.path.abspath(folders.val_dataset_root)
        train_line = f'train: {train_abspath}'
        val_line = f'val: {val_abspath}'
        nc_line = 'nc: 1'
        names_line = 'names: ["car"]'
        file.write(f'{train_line}\n{val_line}\n{nc_line}\n{names_line}')

def main():
    args = parse_args()
    conf = ConfigParser()
    conf.read('config/conf.ini')
    split = read_split(conf)
    folders = make_foldernames(args.images, args.labels, args.output_dataset)
    make_folders(folders)
    process_folders(folders, split)
    write_yaml(folders)
    
if __name__ == '__main__':
    main()