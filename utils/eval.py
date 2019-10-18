import os
import sys
from client.mifid_demo import MIFID
from glob import glob
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


def evaluate(img_dir):
    img_paths = glob(img_dir)
    print('len of image list {}'.format(len(img_paths)))
    mifid = MIFID(model_path='./client/motorbike_classification_inception_net_128_v4_e36.pb',
                  public_feature_path='./client/public_feature.npz')

    img_np = np.empty((len(img_paths), 128, 128, 3), dtype=np.uint8)
    for idx, path in tqdm(enumerate(img_paths)):
        img_arr = cv2.imread(path)[..., ::-1]
        img_arr = np.array(img_arr)
        img_np[idx] = img_arr

    return mifid.compute_mifid(img_np)
