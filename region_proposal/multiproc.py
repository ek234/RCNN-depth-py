import os
import pickle
import numpy as np
import multiprocessing
from utils.extract_features import get_features_from_filename

NUM_PROCESSES = 10
NUM_IMAGES = 1449

img_arr = [i for i in range(NUM_IMAGES)]

def compute_xy(i) :
    return get_features_from_filename(f"{i}")

def run_code(img) :
    X, Y = compute_xy(img)
    # num_half = len(Y) // 2
    # nullids = Y == 0
    # if np.sum(nullids) > num_half:
    #     X = X[not nullids] + X[nullids][:num_half]
    #     Y = Y[not nullids] + Y[nullids][:num_half]
    pickle.dump((X, Y), open(f"./picks/{img}.p", 'wb'))

if __name__ == "__main__" :
    pool = multiprocessing.Pool(processes=NUM_PROCESSES)
    pool.map(run_code, img_arr)
    pool.close()
    pool.join()
