from model.utils.bbox_tools import bbox_iou
import numpy as np

def test_bbox_iou():
    bbox_a = np.ndarray((10, 4), dtype=np.float32)
    bbox_a = np.random.randint(low=0, high=1000, size=(10, 4))
    bbox_b = np.random.randint(low=0, high=1000, size=(20, 4))
    print(bbox_a[:, None, :2].shape)
    print(bbox_b[:, 2:].shape)
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, 2:])
    print(tl.shape)
    print(tl)
    return

test_bbox_iou()