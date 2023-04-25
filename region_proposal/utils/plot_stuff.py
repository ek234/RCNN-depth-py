import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def plot_images ( image, image_depth, image_hha ) :
    _, subplts = plt.subplots(1, 3, figsize=(10, 10))
    subplts[0].imshow(image)
    subplts[0].set_title("RGB")
    subplts[0].axis('off')
    subplts[1].imshow(image_depth)
    subplts[1].set_title("Depth")
    subplts[1].axis('off')
    subplts[2].imshow(image_hha)
    subplts[2].set_title("HHA")
    subplts[2].axis('off')
    plt.show()

def plot_hha_components ( image_hha ) :
    _, subplts = plt.subplots(1, 3, figsize=(10, 10))
    subplts[0].imshow(image_hha[:, :, 0])
    subplts[0].set_title("Angle with gravity")
    subplts[0].axis('off')
    subplts[1].imshow(image_hha[:, :, 1])
    subplts[1].set_title("Height above ground")
    subplts[1].axis('off')
    subplts[2].imshow(image_hha[:, :, 2])
    subplts[2].set_title("Disparity")
    subplts[2].axis('off')
    plt.show()

def plot_angle_directions ( image_hha ) :
    up = image_hha[:, :, 0] < 255/3
    horiz = ( 255/3 <= image_hha[:, :, 0] ) & ( image_hha[:, :, 0] < 2*255/3 )
    down = 2*255/3 <= image_hha[:, :, 0]

    _, subplts = plt.subplots(1, 3, figsize=(10, 10))
    subplts[0].imshow(down)
    subplts[0].set_title("Down")
    subplts[0].axis('off')
    subplts[1].imshow(horiz)
    subplts[1].set_title("Horizontal")
    subplts[1].axis('off')
    subplts[2].imshow(up)
    subplts[2].set_title("Up")
    subplts[2].axis('off')
    plt.show()

def plot_masks ( allmasks, am_label, am_instance ) :
    masks_per_row = 5
    _, subplts = plt.subplots(int(np.ceil(len(allmasks)/masks_per_row)), masks_per_row, figsize=(20, 10))
    for subplt in subplts.flatten() :
        subplt.axis('off')
    for idx, (l, i, imask) in enumerate(zip(am_label, am_instance, allmasks)) :
        subplts[idx//masks_per_row][idx%masks_per_row].imshow(imask)
        subplts[idx//masks_per_row][idx%masks_per_row].axis('off')
        subplts[idx//masks_per_row][idx%masks_per_row].set_title(f"{l}, {i}")
    plt.show()

def showSeSeBBoxes ( image, rects ) :
	bocses_per = 30
	for i in range(0, len(rects), bocses_per):
		output = image.copy()
		for (x, y, w, h) in rects[i:i + bocses_per]:
			color = [random.randint(0, 255) for _ in range(0, 3)]
			cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
		cv2.imshow("Output", output)
		key = cv2.waitKey(0) & 0xFF
		if key == ord("q"):
			break
	cv2.destroyWindow("Output")

def plot_selected_regions ( image, regions_supressed, Y_supressed, bigshow:bool=False, scores=None ) :
    if scores is not None :
        sortingorder = np.argsort(scores)[::-1]
        regions_supressed, Y_supressed = regions_supressed[sortingorder], Y_supressed[sortingorder]
    if not bigshow :
        pics_per_row = 4
        _, subplts = plt.subplots(int(np.ceil(np.sum(Y_supressed!=0)/pics_per_row)), pics_per_row, figsize=(20, 20))
        for subplt in subplts.flatten() :
            subplt.axis('off')

    for idx, ids in enumerate(np.argwhere(Y_supressed != 0).flatten()) :
        x, y, w, h = regions_supressed[ids]
        output = np.zeros_like(image)
        output[y:y+h, x:x+w] = image[y:y+h, x:x+w]
        if not bigshow :
            subplts[idx//pics_per_row][idx%pics_per_row].imshow(output)
            subplts[idx//pics_per_row][idx%pics_per_row].set_title(Y_supressed[ids])
        else :
            plt.imshow(output)
            plt.title(Y_supressed[ids])
            plt.axis('off')
            plt.show()
    plt.show()

def plot_selected_regions_together ( image, regions, Y ) :
    output = image.copy()
    for ids in np.argwhere(Y != 0).flatten() :
        x, y, w, h = regions[ids]
        color = [random.randint(0, 255) for _ in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)
    plt.imshow(output)
    plt.axis('off')
    plt.show()