import pandas as pd
import matplotlib.pyplot as plt

detr_models = ['detr-resnet-50', 'detr-resnet-101']
yolo_models = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
# Avg. MOTA
for model in detr_models:
    df = pd.read_csv(f'{model}_mota.csv')
    c_t = df['conf_threshold']
    mota = df['mota']
    plt.plot(c_t, mota, label=model)
plt.title('Average MOTA')
plt.grid(visible=True)
plt.legend(loc='upper left')
plt.xlabel('Confidence threshold')
plt.ylabel('Average MOTA')
plt.savefig(f'detr_mota.png')
plt.clf()

for model in yolo_models:
    df = pd.read_csv(f'{model}_mota.csv')
    c_t = df['conf_threshold']
    mota = df['mota']
    plt.plot(c_t, mota, label=model)
plt.title('Average MOTA')
plt.grid(visible=True)
plt.legend(loc='upper left')
plt.xlabel('Confidence threshold')
plt.ylabel('Average MOTA')
plt.savefig(f'yolo_mota.png')
plt.clf()

# Maximum:
for model in detr_models:
    df = pd.read_csv(f'{model}_mota.csv')
    c_t = df['conf_threshold']
    mota = df['max_mota']
    plt.plot(c_t, mota, label=model)
plt.title('Maximum MOTA')
plt.grid(visible=True)
plt.legend(loc='upper left')
plt.xlabel('Confidence threshold')
plt.ylabel('Maximum MOTA')
plt.savefig(f'detr_max_mota.png')
plt.clf()

for model in yolo_models:
    df = pd.read_csv(f'{model}_mota.csv')
    c_t = df['conf_threshold']
    mota = df['max_mota']
    plt.plot(c_t, mota, label=model)
plt.title('Maximum MOTA')
plt.grid(visible=True)
plt.legend(loc='upper left')
plt.xlabel('Confidence threshold')
plt.ylabel('Maximum MOTA')
plt.savefig(f'yolo_max_mota.png')
plt.clf()