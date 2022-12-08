import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    models = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', 'detr-resnet-50', 'detr-resnet-101']
    # Read and join curves
    P = pd.read_csv('p_curve.csv', index_col=0)
    R = pd.read_csv('r_curve.csv', index_col=0)
    PR = P.join(R, lsuffix='_P', rsuffix='_R')
    # create pr_mota df
    total_pr_mota = pd.DataFrame(columns=models, index=['pr_mota'])
    # plot
    for m_n in models:
        mota = pd.read_csv(f'{m_n}_mota.csv', index_col='conf_threshold')
        pr_mota = PR.join(mota)
        p_prev, r_prev, mota_prev = pr_mota[[f'{m_n}_P', f'{m_n}_R', 'mota']].iloc[0]
        integral = 0.0
        for conf_t, row in pr_mota.iloc[1:].iterrows():
            p, r, mota = row[[f'{m_n}_P', f'{m_n}_R', 'mota']]
            ds = np.linalg.norm([p-p_prev, r-r_prev])
            sum_trapezoid = mota_prev + mota
            integral += 0.5*(sum_trapezoid*ds)
            p_prev, r_prev, mota_prev = p, r, mota
        # add fictive last point: conf threshold = 1.0, no detections, p = 1, r = 0
        integral += 0.5 * np.linalg.norm([1-p_prev, 0-r_prev]) * mota_prev
        total_pr_mota[m_n] = 0.5*integral
    # write:
    total_pr_mota.to_csv('pr_mota_0.5.csv')

            
            

if __name__ == '__main__':
    main()