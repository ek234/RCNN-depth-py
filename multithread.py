import os
import multiprocessing
import pickle

NUM_PROCESSES = 2


img_arr = [i for i in range(1500)]
data = {i: {"X": [], "Y": []} for i in img_arr}

# count = 0

def compute_xy(i):
    print("pid", os.getpid())
    return f"X_{i}", f"Y_{i}"

def run_code(img):
    X, Y = compute_xy(img)
    data[img]["X"] = X
    data[img]["Y"] = Y

print("computation done..")

print("Writing to pickle...")
pickle.dump(data, open("save_xy.p", "wb"))

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=NUM_PROCESSES)
    pool.map(run_code, img_arr)

    pool.close()
    pool.join()

    print("GG")
