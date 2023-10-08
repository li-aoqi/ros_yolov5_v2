# ros_yolov5_v2

订阅图像话题image_topic进行yolo识别，然后将置信度最高的那个矩形框的中心坐标以PoseStamped的数据格式发送出去result_topic

## 00 工作环境配置

```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone git@github.com:li-aoqi/ros_yolov5_v2.git
```

## 01 环境配置

```
cd ros_yolov5_v2
pip3 install -r requirements.txt
```

## 02 更改你所需要的图像话题的名称

```python
#laq_debug_cpu.py
image_topic = "/camera/color/image_raw" 
```

## 03 编译

```
cd ~/catkin_ws
catkin_make
```

## 04 运行

```
source devel/setup.bash
rosrun ros_yolov5_v2 laq_debug_cpu.py
```

