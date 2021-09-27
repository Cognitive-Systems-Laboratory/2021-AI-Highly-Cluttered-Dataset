from PIL import Image
import pprint
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import random
import time
import os
import copy
from collections import Counter

from dataset import coco_train, coco_valid, COCO_INSTANCE_CATEGORY_NAMES

colors = ((0, 255, 0),
           (0, 0, 255),
           (255, 0, 0),
           (0, 255, 255),
           (255, 255, 0),
           (255, 0, 255),
           (80, 70, 180),
           (250, 80, 190),
           (245, 145, 50),
           (70, 150, 250),
           (50, 190, 190))


def random_colour_masks(image):
    """
    random_colour_masks
    parameters:
      - image - predicted masks
    method:
      - the masks of each predicted object is given random colour for visualization
    """
    idx = random.randrange(0,10)
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colors[idx]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask, colors[idx]

def get_prediction(model, img_path, threshold, method, show_log=True):
    masks = None

    img = Image.open(img_path)  # Load the image
    transform = T.Compose([T.ToTensor()])  # Defing PyTorch Transform
    img = transform(img)  # Apply the transform to the image
    pred = model([img])  # Pass the image to the model

    if show_log:
        print(f"Model: {type(model).__name__}")
        print(f"Mode: {method}")
        print(f"Threshold: {threshold}")
        print(f"Image path: {img_path}")

    imgno = int(img_path[-16:-4])
    cats = coco_train.loadCats(coco_train.getCatIds())
    imgIds = coco_train.getImgIds(imgIds = [imgno]) 
    catIds = coco_train.getCatIds(catNms=[cat['name'] for cat in cats])
    annIds = coco_train.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=False)
    anns = coco_train.loadAnns(annIds)
    
    if method == 'detection':
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
        # pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]  
        pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(pred[0]['boxes'].detach().numpy())]
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
        
        if not pred_t:
            return None, []
        
        pred_t = pred_t[-1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
    
    elif method == 'segmentation':

        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
        masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(pred[0]['boxes'].detach().numpy())]
        masks = masks[:pred_t+1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]

    if show_log:
        print(f"Ground truth: {len(anns)} objects / Detected {len(pred_boxes)} objects (threshold:{threshold})")
        print('Ground truth: ')
        pprint.pprint(Counter([COCO_INSTANCE_CATEGORY_NAMES[ann['category_id']] for ann in anns]), width=1)
        print('Predicted: ')
        pprint.pprint(Counter(pred_class), width=1)

    return anns, masks, pred_boxes, pred_class


def object_detection_api(model, img_path, threshold=0.8, rect_th=1, text_size=1, text_th=1):
    anns, _, pred_boxes, pred_cls = get_prediction(model, img_path, threshold, 'detection') # Get predictions

    img = cv2.imread(img_path) # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB

    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    img_origin = copy.deepcopy(img)

    gt_boxes = list(map(lambda x: [(int(x[0]), int(x[1])),
                                   (int(x[0]+x[2]), int(x[1]+x[3]))], [ann['bbox'] for ann in anns]))
    for ii in range(len(gt_boxes)):
        # Draw Rectangle with the coordinates
        cv2.rectangle(img_origin, gt_boxes[ii][0], gt_boxes[ii][1], 
                      color=(255, 255, 255), thickness=2)

    plt.imshow(img_origin)
    plt.axis('off')
    plt.title('Ground Truth')

    plt.subplot(1, 2, 2)

    for i in range(len(pred_boxes)):
        idx = random.randrange(0,10)
        # Draw Rectangle with the coordinates
        cv2.rectangle(img, pred_boxes[i][0], pred_boxes[i][1],color=colors[idx], thickness=rect_th) 
        # Write the prediction class
        cv2.putText(img, f"[{i:2d}]"+pred_cls[i], pred_boxes[i][0],
                    cv2.FONT_HERSHEY_SIMPLEX, text_size, color=colors[idx],thickness=text_th) 
    plt.imshow(img)
    plt.title('Predicted')
    plt.axis('off')

    # plt.suptitle(f"Image: {img_path} / Ground truth: {len(anns)} objects / Detected {len(pred_boxes)} objects (threshold:{threshold})")
    plt.show()

def instance_segmentation_api(model, img_path, threshold=0.8, rect_th=1, text_size=1, text_th=1):
    """
    instance_segmentation_api
    parameters:
      - img_path - path to input image
    method:
      - prediction is obtained by get_prediction
      - each mask is given random color
      - each mask is added to the image in the ration 1:0.8 with opencv
      - final output is displayed
    """
    anns, masks, pred_boxes, pred_cls = get_prediction(model, img_path, threshold, 'segmentation')

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    img_origin = copy.deepcopy(img)

    plt.imshow(img_origin)

    for ann in anns:
        # Draw Rectangle with the coordinates
        plt.fill(ann['segmentation'][0][0::2], ann['segmentation'][0][1::2], alpha=0.8)
    
    plt.axis('off')
    plt.title('Ground Truth')

    plt.subplot(1, 2, 2)

    for i in range(len(masks)):
        rgb_mask, colours = random_colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, pred_boxes[i][0], pred_boxes[i][1],color=colours, thickness=rect_th)
        cv2.putText(img, f"[{i:2d}]"+pred_cls[i], pred_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, colours, thickness=text_th)

    plt.imshow(img)
    plt.axis('off')
    plt.title('Predicted')
    # plt.suptitle(f"Image: {img_path} / Ground truth: {len(anns)} objects / Detected {len(pred_boxes)} objects (threshold:{threshold})")
    plt.show()


if __name__ == '__main__':
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()