# encoding: utf-8
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2


def visualize_detection(img, dets, is_show_label=True, classes=None,
                        thresh=0.5):
    """
    visualize detections in one image

    Parameters:
    ----------
    img : numpy.array image, in bgr format
    dets : numpy.array ssd detections,
            numpy.array([[x1, y1, x2, y2, score, cls_id]...])
    classes : tuple or list of str class names
    thresh : float, score threshold
    """
    plt.imshow(img)
    colors = dict()
    for det in dets:
        bb = det[:4].astype(int)

        if is_show_label:
            cls_id = int(det[5])
            score = det[4]
            if cls_id == 0:
                continue
            if score > thresh:
                if cls_id not in colors:
                    colors[cls_id] = (random.random(), random.random(),
                                      random.random())
                rect = plt.Rectangle((bb[0], bb[1]), bb[2] - bb[0],
                                     bb[3] - bb[1], fill=False,
                                     edgecolor=colors[cls_id],
                                     linewidth=3.5)
                plt.gca().add_patch(rect)
                if classes and len(classes) > cls_id:
                    cls_name = classes[cls_id]
                else:
                    cls_name = str(cls_id)
                plt.gca().text(bb[0], bb[1] - 2,
                               '{:s} {:.3f}'.format(cls_name, score),
                               bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                               fontsize=12, color='white')
        else:
            rect = plt.Rectangle((bb[0], bb[1]), bb[2] - bb[0],
                                 bb[3] - bb[1], fill=False,
                                 edgecolor=(1, 0, 0),
                                 linewidth=3.5)
            plt.gca().add_patch(rect)
    plt.show()


# visualize_old: use opencv api
def visualize_detection_old(img, dets, is_show_label=True, classes=None,
                            thresh=0.5, name='detection'):
    """
    visualize detections in one image

    Parameters:
    ----------
    img : numpy.array image, in bgr format
    dets : numpy.array ssd detections,
            numpy.array([[x1, y1, x2, y2, score, cls_id]...])
    classes : tuple or list of str class names
    thresh : float, score threshold
    """
    im = np.array(img)
    colors = dict()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in dets:
        bb = det[:4].astype(int)
        if is_show_label:
            cls_id = int(det[5])
            score = det[4]

            if cls_id == 0:
                continue
            if score > thresh:
                if cls_id not in colors:
                    colors[cls_id] = (
                        random.random() * 255, random.random() * 255,
                        random.random() * 255)

                cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]),
                              colors[cls_id], 3)

                if classes and len(classes) > cls_id:
                    cls_name = classes[cls_id]
                else:
                    cls_name = str(cls_id)
                cv2.putText(im, '{:s} {:.3f}'.format(cls_name, score),
                            (bb[0], bb[1] - 2),
                            font, 0.5, colors[cls_id], 1)
        else:
            cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)

    cv2.imshow(name, im)
    while True:
        c = cv2.waitKey(100000)
        if c == ord('d'):
            return
        elif c == ord('n'):
            break
