#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
import time

import argparse
import numpy as np
from PIL import Image
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.pytorch_loader import load_pytorch_model, pytorch_inference

# Ros libraries
import roslib
import rospy
# Ros Messages
from sensor_msgs.msg import CompressedImage


class MaskDetector:
    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.image_pub = rospy.Publisher("/output/face_mask",
                                          CompressedImage)
        # self.bridge = CvBridge()
        # subscribed Topic
        self.subscriber = rospy.Subscriber("/camera_out/image",
                                            CompressedImage, self._callback,  queue_size = 1)
        #self.model = load_pytorch_model('models/face_mask_detection.pth');
        self.model = load_pytorch_model('/ari_public_ws/src/FaceMaskDetection/models/model360.pth');
        # anchor configuration
        #feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
        feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
        anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
        anchor_ratios = [[1, 0.62, 0.42]] * 5
        
        # generate anchors
        anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
        
        # for inference , the batch size is 1, the model output shape is [1, N, 4],
        # so we expand dim for anchors to [1, anchor_num, 4]
        self.anchors_exp = np.expand_dims(anchors, axis=0)
        self.id2class = {0: 'Mask', 1: 'NoMask'}

    def _callback(self, ros_data):
        ''' callback function for mask detection '''
        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        #image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:

        boxes, image = self.inference(image_np)


        #### Create Compressed Image ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)
        #self.subscriber.unregister()


    def inference(self,
                  image,
                  conf_thresh=0.5,
                  iou_thresh=0.4,
                  target_shape=(160, 160),
                  draw_result=True,
                  show_result=True
                  ):
        '''
        Main function of detection inference
        :param image: 3D numpy array of image
        :param conf_thresh: the min threshold of classification probabity.
        :param iou_thresh: the IOU threshold of NMS
        :param target_shape: the model input size.
        :param draw_result: whether to daw bounding box to the image.
        :param show_result: whether to display the image.
        :return:
        '''
        # image = np.copy(image)
        output_info = []
        height, width, _ = image.shape
        image_resized = cv2.resize(image, target_shape)
        image_np = image_resized / 255.0  # 归一化到0~1
        image_exp = np.expand_dims(image_np, axis=0)
    
        image_transposed = image_exp.transpose((0, 3, 1, 2))
    
        y_bboxes_output, y_cls_output = pytorch_inference(self.model, image_transposed)
        # remove the batch dimension, for batch is always 1 for inference.
        y_bboxes = decode_bbox(self.anchors_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]
        # To speed up, do single class NMS, not multiple classes NMS.
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)
    
        # keep_idx is the alive bounding box after nms.
        keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                     bbox_max_scores,
                                                     conf_thresh=conf_thresh,
                                                     iou_thresh=iou_thresh,
                                                     )
        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            class_id = bbox_max_score_classes[idx]
            bbox = y_bboxes[idx]
            # clip the coordinate, avoid the value exceed the image boundary.
            xmin = max(0, int(bbox[0] * width))
            ymin = max(0, int(bbox[1] * height))
            xmax = min(int(bbox[2] * width), width)
            ymax = min(int(bbox[3] * height), height)
    
            if draw_result:
                if class_id == 0:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(image, "%s: %.2f" % (self.id2class[class_id], conf), (xmin + 2, ymin - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
            output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
    
        return [output_info, image]

if __name__ == "__main__":
    detector = MaskDetector()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")


