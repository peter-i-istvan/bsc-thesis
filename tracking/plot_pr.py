import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Read and join curves
    P = pd.read_csv('p_curve.csv', index_col=0)
    R = pd.read_csv('r_curve.csv', index_col=0)
    PR = P.join(R, lsuffix='_P', rsuffix='_R')
    # plot
    for m_n in ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', 'detr-resnet-50', 'detr-resnet-101']:
        ps, rs = [], []
        for conf_threshold, row in  PR.iterrows():
            p = row[f'{m_n}_P']
            r = row[f'{m_n}_R']
            ps.append(p)
            rs.append(r)
        plt.plot(ps, rs)
        plt.xlabel('Precision')
        plt.xlim((-0.1, 1.1))
        plt.ylabel('Recall')
        plt.ylim((-0.1, 1.1))
        plt.title(f'PR curve of {m_n}')
        plt.savefig(f'{m_n}.png')
        plt.clf()

            

if __name__ == '__main__':
    main()