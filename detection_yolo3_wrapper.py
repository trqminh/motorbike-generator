"""
aioz.aiar.truongle - june 06, 2019
yolov3 object detection
"""
import numpy as np
import tensorflow as tf
import cv2
import glob
from PIL import Image


class ObjectDetection:
    def __init__(self, graph_path="./detection_yolov3.pb", im_size=416, threshold=0.9, memory_fraction=0.3):
        # config gpu
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
        self.config.gpu_options.allow_growth = True
        self.config.log_device_placement = False

        self.graph_fp = graph_path
        self.im_size = im_size
        self.threshold = threshold
        self.session = None

        self._load_graph()
        self._init_predictor()

    def _load_graph(self):
        print('[INFO] Load graph at {} ... '.format(self.graph_fp))
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_fp, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        # tf.get_default_graph().finalize()

    def _init_predictor(self):
        # print('[INFO] Init predictor ...')
        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph, config=self.config)
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes_tensor = self.graph.get_tensor_by_name('concat_10:0')
            self.score_tensor = self.graph.get_tensor_by_name('concat_11:0')
            self.labels_tensor = self.graph.get_tensor_by_name('concat_12:0')

    def process_prediction(self, image):

        height_ori, width_ori = image.shape[:2]
        img = cv2.resize(image, (self.im_size, self.im_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        boxes_, scores_, labels_ = self.session.run([self.boxes_tensor, self.score_tensor, self.labels_tensor],
                                                    feed_dict={self.image_tensor: img})

        # filter
        indices = np.where(scores_ > self.threshold)
        boxes = boxes_[indices]
        labels = labels_[indices]

        # rescale the coordinates to the original image
        boxes[:, 0] *= (width_ori / float(self.im_size))
        boxes[:, 2] *= (width_ori / float(self.im_size))
        boxes[:, 1] *= (height_ori / float(self.im_size))
        boxes[:, 3] *= (height_ori / float(self.im_size))

        # check boxes
        idi = np.where(boxes[:, 0] < 0)[0]
        boxes[idi, 0] = 0
        idi = np.where(boxes[:, 1] < 0)[0]
        boxes[idi, 1] = 0
        idi = np.where(boxes[:, 2] > width_ori)[0]
        boxes[idi, 2] = width_ori
        idi = np.where(boxes[:, 3] > height_ori)[0]
        boxes[idi, 3] = height_ori

        return boxes.astype(int), labels


def check_white_bg(roi):
    item = roi[0, 0]
    item1 = roi[0, 1]
    item2 = roi[1, 0]
    item3 = roi[1, 1]
    if not (item[0] == 255 and item[1] == 255 and item[2] == 255):
        return False
    
    if not (item1[0] == 255 and item1[1] == 255 and item1[2] == 255):
        return False
    
    if not (item2[0] == 255 and item2[1] == 255 and item2[2] == 255):
        return False
    
    if not (item3[0] == 255 and item3[1] == 255 and item3[2] == 255):
        return False

    return True

def crop_obj_from_im_folder(folder_path):
    det = ObjectDetection()
    path = folder_path + "*"
    num_img = 0
    num_fail = 0
    for file in glob.glob(path):
        im = cv2.imread(file)
        
        try:
            boxes, labels = det.process_prediction(im)
            for i, label in enumerate(labels):
                if label == 3: 
                    roi = im[boxes[i][1]:boxes[i][3], boxes[i][0]:boxes[i][2]]
                    height, width, channels = roi.shape
                    item = roi[0, 0]
                    if width*1.4 < height:
                        roi = cv2.resize(roi, (128, 128))
                        file_name = str(num_img) + '.png'
                        cv2.imwrite('./not_sure_if_clean_img/1/' + file_name, roi)  # front of motor
                        num_img += 1
                    elif check_white_bg(roi):
                        roi = cv2.resize(roi, (128, 128))
                        file_name = str(num_img) + '.png'
                        cv2.imwrite('./not_sure_if_clean_img/0/' + file_name, roi)  # motor with white bg
                        num_img += 1
                    elif width < 128 or height < 128:
                        # roi = cv2.resize(roi, (128, 128))
                        # file_name = str(num_img) + '.png'
                        # cv2.imwrite('./not_sure_if_clean_img/2/' + file_name, roi)   # small one
                        # num_img += 1
                        print('maybe noise')
                    else:
                        roi = cv2.resize(roi, (128, 128))
                        file_name = str(num_img) + '.png'
                        cv2.imwrite('./not_sure_if_clean_img/3/' + file_name, roi)  # motor and things
                        num_img += 1

        except:
            num_fail+=1

        if num_img % 1000 == 0:
            print('Process {} images'.format(num_img))
        
    print('fail: ', num_fail)    
        
        


def main():
    crop_obj_from_im_folder('/home/aioz-interns/Downloads/data/training_dataset/motobike/')

if __name__ == "__main__":
    main()