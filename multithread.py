import concurrent.futures
import pickle
from region_proposal.utils.extract_features import get_features_from_filename

NUM_WORKERS = 4

pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
img_arr = [[3], [5], [6], [2]]
flat = [item for sublist in img_arr for item in sublist]
data = {i: {"X": [], "Y": []} for i in flat}

count = 0

def compute_xy(i):
    return get_features_from_filename(f"{i}")

def run_code(arr):
    global count
    for img in arr:
        X, Y = compute_xy(img)
        print("count", count)
        data[img]["X"] = X
        data[img]["Y"] = Y
        count += 1

for i in range(NUM_WORKERS):
    pool.submit(run_code, img_arr[i])

pool.shutdown(wait=True)
print("computation done..")

print("Writing to pickle...")
pickle.dump(data, open("save_xy.p", "wb"))
