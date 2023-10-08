#!/usr/bin/env python3
import numpy as np
import cv2
import random
import torch
import time
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
import os

#加载本地的模型文件

# 可以使用自己训练的模型来替换
model_weights_path = os.getcwd()+'/src/ros_yolov5_v2/src'
pt_path = os.getcwd()+'/src/ros_yolov5_v2/src/best_circle.pt'
model = torch.hub.load(model_weights_path, 'custom',pt_path, source='local')  # local repo
# model = torch.hub.load(model_weights_path, 'custom','best_car.pt', source='local')  # local repo
bridge = CvBridge()

def dectshow(org_img, boxs):
    img = org_img.copy()
    for box in boxs:
        # print(box)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(img, box[-1]+'('+str(int((boxs[0][0]+boxs[0][2])/2))+','+str(int(boxs[0][1]+boxs[0][3])/2)+')',
                    (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.putText(img, box[-1],
        #             (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('detect_img', img)
    cv2.waitKey(1)

def detect_image_callback(msg):
    try:
        # 将ROS图像消息转换为OpenCV格式
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # 在这里可以对cv_image进行进一步处理
        # 调节图片的大小
        cv_image = cv2.resize(cv_image,(640,480))
        # 转变为yolo所需要的格式
        color_image = np.asanyarray(cv_image)

        results = model(color_image)
        boxs= results.pandas().xyxy[0].values

        # boxs= results.pandas().xyxy[0].values
        if(len(boxs)==0):
            print("[error]: 一个目标都没有识别到")
        else:
            position_x = int((boxs[0][0]+boxs[0][2])/2)
            position_y = int((boxs[0][1]+boxs[0][3])/2)
            print("置信度最高的中心点的坐标为:",(position_x,position_y))
            position_msg = PoseStamped()
            position_msg.header.stamp=rospy.Time.now()
            position_msg.pose.position.x = position_x
            position_msg.pose.position.y = position_y
            position_msg.pose.position.z = 0
            result_publisher.publish(position_msg)
        dectshow(color_image, boxs)

    except CvBridgeError as e:
        rospy.logerr(e)


if __name__ == '__main__':

    rospy.init_node('ros_yolov5_node')
    # image_topic = "/camera/color/image_raw"  # 图像话题
    # image_topic = "/prometheus/sensor/monocular_front/image_raw"
    image_topic = "/airsim_node/drone_1/front_left/Scene"

    result_topic = "/object_detection_results"  # 结果话题

    # 创建一个发布器，发布检测结果//置信度最高的图像信息x,y
    result_publisher = rospy.Publisher(result_topic, PoseStamped, queue_size=10)
    # 创建一个订阅器，订阅图像话题
    rospy.Subscriber(image_topic, Image, detect_image_callback)
    rospy.spin()
