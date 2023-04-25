import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.plot_stuff import plot_images, plot_hha_components, plot_angle_directions, plot_masks, showSeSeBBoxes, plot_selected_regions

MIN_SIMILARITY_SCORE = 0.5 # Minimum score needed to be considered not garbage
MIN_SIMILARITY_WITH_NEIGHBOR_BBOX = 0.5 # Minimum score needed to be considered the same as a neighbor region
NeedFastSeSe = False


def getCameraParams ( isColor: bool ) :
    # TODO : consider distortion

    # RGB Intrinsic Parameters
    fx_rgb = 5.1885790117450188e+02
    fy_rgb = 5.1946961112127485e+02
    cx_rgb = 3.2558244941119034e+02
    cy_rgb = 2.5373616633400465e+02
    # # RGB Distortion Parameters
    # k1_rgb =  2.0796615318809061e-01
    # k2_rgb = -5.8613825163911781e-01
    # k3_rgb = 4.9856986684705107e-01
    # p1_rgb = 7.2231363135888329e-04
    # p2_rgb = 1.0479627195765181e-03

    # Depth Intrinsic Parameters
    fx_d = 5.8262448167737955e+02
    fy_d = 5.8269103270988637e+02
    cx_d = 3.1304475870804731e+02
    cy_d = 2.3844389626620386e+02
    # # RGB Distortion Parameters
    # k1_d = -9.9897236553084481e-02
    # k2_d = 3.9065324602765344e-01
    # k3_d = -5.1031725053400578e-01
    # p1_d = 1.9290592870229277e-03
    # p2_d = -1.9422022475975055e-03

    if isColor :
        return cx_rgb, cy_rgb, fx_rgb, fy_rgb
    return cx_d, cy_d, fx_d, fy_d


def getImages ( img_name: str, root_dir: str = "../nyudv2", print_info: bool = False ) :
    start = time.time()

    image = np.load(f"{root_dir}/rgb/{img_name}.npy")
    image_depth = np.load(f"{root_dir}/depth/{img_name}.npy")
    image_labelmaps = np.load(f"{root_dir}/label/{img_name}.npy")
    image_instmaps = np.load(f"{root_dir}/instance/{img_name}.npy")
    image_hha = np.load(f"{root_dir}/hha/{img_name}.npy")

    image_xyz = np.zeros((image_depth.shape[0], image_depth.shape[1], 3))
    height, width = image_depth.shape
    CX_DEPTH, CY_DEPTH, FX_DEPTH, FY_DEPTH = getCameraParams(isColor=False)
    for i in range(height):
        for j in range(width):
            z = image_depth[i,j]
            x = (j - CX_DEPTH) * z / FX_DEPTH
            y = (i - CY_DEPTH) * z / FY_DEPTH
            image_xyz[i,j] = [x, y, z]

    image_labinsts = np.concatenate((image_labelmaps[:, :, np.newaxis], image_instmaps[:, :, np.newaxis]), axis=-1)
    fg_linsts = np.array(list({ (l,i) for l, i in image_labinsts.reshape(-1, 2) if l != 0 and i != 0 })) # 0 is background (boundaries, etc)

    allmasks, am_label, am_instance, am_bbox = list(), list(), list(), list()
    for label, inst in fg_linsts :
        mask = (image_labinsts[:, :, 0] == label) & (image_labinsts[:, :, 1] == inst)
        allmasks.append(mask)
        am_label.append(label)
        am_instance.append(inst)
        xmin, xmax, ymin, ymax = mask.shape[1], 0, mask.shape[0], 0
        for y in range(mask.shape[0]) :
            for x in range(mask.shape[1]) :
                if mask[y, x] :
                    ymin = min(ymin, y)
                    ymax = max(ymax, y)
                    xmin = min(xmin, x)
                    xmax = max(xmax, x)
        assert ymin <= ymax
        assert xmin <= xmax
        am_bbox.append(( xmin, ymin, xmax-xmin, ymax-ymin ))
    allmasks = np.array(allmasks)
    am_label = np.array(am_label)
    am_instance = np.array(am_instance)
    am_bbox = np.array(am_bbox)

    if print_info :
        unilm = np.unique(image_labelmaps)
        print(f"[INFO] unique label classes: {unilm}")
        print(f"[INFO] number of unique label classes: {len(unilm)}")
        print(f"[INFO] all images loaded in {time.time() - start:.4f}s")

    # ensure all of these are np arrays
    return image, image_depth, image_hha, image_xyz, image_labinsts, allmasks, am_label, am_instance, am_bbox


def performSeSe ( image, image_hha, print_info=False ) :
	start = time.time()
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(image)
	ss.addImage(image_hha) # TODO : test reverse order
	if NeedFastSeSe :
		if print_info :
			print("[INFO] using *fast* selective search")
		ss.switchToSelectiveSearchFast()
	else:
		if print_info :
			print("[INFO] using *quality* selective search")
		ss.switchToSelectiveSearchQuality()
	bboxes = ss.process()
	if print_info :
		print(f"[INFO] total number of region proposals: {len(bboxes)}")
		print(f"[INFO] basic region proposal took {time.time()-start:.4f} seconds")
	return bboxes


def performSeSe ( image, image_hha, print_info=False ) :
	start = time.time()
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(image)
	ss.addImage(image_hha) # TODO : test reverse order
	if NeedFastSeSe :
		if print_info :
			print("[INFO] using *fast* selective search")
		ss.switchToSelectiveSearchFast()
	else:
		if print_info :
			print("[INFO] using *quality* selective search")
		ss.switchToSelectiveSearchQuality()
	bboxes = ss.process()
	if print_info :
		print(f"[INFO] total number of region proposals: {len(bboxes)}")
		print(f"[INFO] basic region proposal took {time.time()-start:.4f} seconds")
	return bboxes


def bb_jaccardsimilarity ( bb1, bb2 ) :
    x1,y1,w1,h1 = bb1
    x2,y2,w2,h2 = bb2
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = min(x1+w1, x2+w2) - x_intersection
    h_intersection = min(y1+h1, y2+h2) - y_intersection

    if w_intersection <= 0 or h_intersection <= 0 :
        return 0.0
    intersection_area = w_intersection * h_intersection
    iou = ( intersection_area ) / ( w1*h1 + w2*h2 - intersection_area )
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def ground_truth_region_label ( query_bbox, bbox_labinsts, am_label, am_instance, am_bbox ) :
    best_class, best_score = 0, 0
    labels_in_question = np.unique(bbox_labinsts[:, :, 0])
    insts_in_question = np.unique(bbox_labinsts[:, :, 1])

    for label, inst, bbox in zip(am_label, am_instance, am_bbox) :
        if label not in labels_in_question or inst not in insts_in_question :
            # TODO : rather than iterating through all possible l,i and
            # checking if that combo is present in this region, do vice versa
            continue
        sim = bb_jaccardsimilarity(bbox, query_bbox)
        if sim > best_score :
            best_score, best_class = sim, label
    return best_class, best_score


def supress_non_maximal_boxes ( rects, image_labinsts, am_label, am_instance, am_bbox, print_info=False ) :
    start = time.time()
    data_unsupressed = list()
    for (x, y, w, h) in rects :
        gt_label, gt_score = ground_truth_region_label((x,y,w,h), image_labinsts[y:y+h, x:x+w], am_label, am_instance, am_bbox)
        data_unsupressed.append((x, y, w, h, gt_label, gt_score))
    data_unsupressed = np.asanyarray(data_unsupressed)
    sorting_order = data_unsupressed[:, -1].argsort()[::-1]
    data_unsupressed = data_unsupressed[sorting_order]
    # NOTE : now the data_unsupressed is sorted by gt_score (descending order)

    if print_info :
        print(f"[INFO] label and score calculation took {time.time()-start:.4f}s")
        start = time.time()

    # supress non maximal boxes
    regions_supressed, labels_supressed, scores_supressed = list(), list(), list()
    for data in data_unsupressed :
        bb = data[:4]
        for bb_ in regions_supressed :
            if bb_jaccardsimilarity(bb, bb_) > MIN_SIMILARITY_WITH_NEIGHBOR_BBOX :
                break
        else :
            # NOTE : this is not a typo, it is a python feature
            # NOTE : code reaches here only if the above for loop is not exited by break
            regions_supressed.append(bb)
            labels_supressed.append(data[4])
            scores_supressed.append(data[5])
    regions_supressed = np.asanyarray(regions_supressed).astype(int)
    labels_supressed = np.asanyarray(labels_supressed).astype(int)
    scores_supressed = np.asanyarray(scores_supressed)

    if print_info :
        print(f"[INFO] number of boxes after supression: {len(labels_supressed)}")
        print(f"[INFO] non-maximum supression took {time.time()-start:.4f}s")
    return regions_supressed, labels_supressed, scores_supressed


def extract_features ( image, image_depth, image_hha, image_xyz, regions_supressed, labels_supressed, scores_supressed, print_info: bool = False ) :
    start = time.time()

    labels_filtered = np.where( scores_supressed < MIN_SIMILARITY_SCORE, 0, labels_supressed )

    X_supressed, Y_supressed = list(), list()
    for (x, y, w, h), gt_label, gt_score, raw_label in zip(regions_supressed, labels_filtered, scores_supressed, labels_supressed) :
        proposed_box_rgb = image[y:y+h, x:x+w]
        proposed_box_depth = image_depth[y:y+h, x:x+w]
        proposed_box_angle = image_hha[y:y+h, x:x+w][:, :, 0]
        proposed_box_height = image_hha[y:y+h, x:x+w][:, :, 1]
        proposed_box_disparity = image_hha[y:y+h, x:x+w][:, :, 2]
        proposed_box_xyz = image_xyz[y:y+h, x:x+w].reshape(-1, 3)

        depth_mean_sd = np.mean(proposed_box_depth), np.std(proposed_box_depth)
        height_mean_sd = np.mean(proposed_box_height), np.std(proposed_box_height)
        angle_mean_sd = np.mean(proposed_box_angle), np.std(proposed_box_angle)
        disparity_mean_sd = np.mean(proposed_box_disparity), np.std(proposed_box_disparity)
        x_mean_sd = np.mean(proposed_box_xyz[:, 0]), np.std(proposed_box_xyz[:, 0])
        y_mean_sd = np.mean(proposed_box_xyz[:, 1]), np.std(proposed_box_xyz[:, 1])
        z_mean_sd = np.mean(proposed_box_xyz[:, 2]), np.std(proposed_box_xyz[:, 2])
        extent_x = np.max(proposed_box_xyz[:, 0]) - np.min(proposed_box_xyz[:, 0])
        extent_y = np.max(proposed_box_xyz[:, 1]) - np.min(proposed_box_xyz[:, 1])
        extent_z = np.max(proposed_box_xyz[:, 2]) - np.min(proposed_box_xyz[:, 2])
        min_height = np.min(proposed_box_height)
        max_height = np.max(proposed_box_height)
        frac_facing_down = np.sum(proposed_box_angle < 255/3) / (w*h)
        frac_facing_horiz = np.sum(( 255/3 <= proposed_box_angle ) & ( proposed_box_angle < 2*255/3 )) / (w*h)
        frac_facing_up = np.sum(2*255/3 <= proposed_box_angle) / (w*h)

        area = w*h
        perimeter = 2*(w+h)
        location = x / image.shape[1], y / image.shape[0]
        aspect_ratio = w / h
        # perimeter (and sum of contour strength) divided by the squared root of the area
        # area of the region divided by that of the bounding box.
        # Sum of contour strength at the boundaries
        # mean contour strength at the boundaries
        # minimum and maximum UCM threshold of appearance and disappearance of the regions forming the candidate.

        r_mean_sd = np.mean(proposed_box_rgb[:, :, 0]), np.std(proposed_box_rgb[:, :, 0])
        g_mean_sd = np.mean(proposed_box_rgb[:, :, 1]), np.std(proposed_box_rgb[:, :, 1])
        b_mean_sd = np.mean(proposed_box_rgb[:, :, 2]), np.std(proposed_box_rgb[:, :, 2])
        
        features = [ raw_label,gt_score,x,y,w,h, *depth_mean_sd, *height_mean_sd, *angle_mean_sd, *disparity_mean_sd, *x_mean_sd, *y_mean_sd, *z_mean_sd, extent_x, extent_y, extent_z, min_height, max_height, frac_facing_down, frac_facing_horiz, frac_facing_up, area, perimeter, *location, aspect_ratio, *r_mean_sd, *g_mean_sd, *b_mean_sd ]
        # NOTE : remove gt_score,x,y,w,h from features before training
        X_supressed.append(features)
        Y_supressed.append(gt_label)
    X_supressed = np.asanyarray(X_supressed)
    Y_supressed = np.asanyarray(Y_supressed)
    
    if print_info :
        print("[INFO] classes and their counts:")
        print(np.asanyarray(np.unique(Y_supressed, return_counts=True)).T)
        print("[INFO] number of classes excluding background : ", len(np.unique(Y_supressed[Y_supressed!=0])))
        print(f"[INFO] feature extraction took {time.time()-start:.4f} seconds")
    return X_supressed, Y_supressed


def get_features_from_filename ( imgnumber, print_plots=False, print_info=False ) :
    tic = time.time()
    image, image_depth, image_hha, image_xyz, image_labinsts, allmasks, am_label, am_instance, am_bbox = getImages(imgnumber, print_info=print_info)
    if print_plots :
        plot_masks(allmasks, am_label, am_instance)
    allbboxes = performSeSe(image, image_hha, print_info=print_info)
    bboxes, labels, scores = supress_non_maximal_boxes(allbboxes, image_labinsts, am_label, am_instance, am_bbox, print_info=print_info)
    X, Y = extract_features(image, image_depth, image_hha, image_xyz, bboxes, labels, scores, print_info=print_info)
    if print_plots :
        plot_selected_regions(image, bboxes, Y)
    if print_info :
        print(f"[INFO] X-> {X.shape} \t Y-> {Y.shape}")
        print(f"[INFO] total time taken {time.time()-tic:.4f} seconds")
    return X, Y


def getImagesBasic ( img_name: str, root_dir: str = "../nyudv2" ) :
    image = np.load(f"{root_dir}/rgb/{img_name}.npy")
    image_depth = np.load(f"{root_dir}/depth/{img_name}.npy")
    image_labelmaps = np.load(f"{root_dir}/label/{img_name}.npy")
    image_instmaps = np.load(f"{root_dir}/instance/{img_name}.npy")
    image_hha = np.load(f"{root_dir}/hha/{img_name}.npy")
    return image, image_depth, image_hha, image_labelmaps, image_instmaps
