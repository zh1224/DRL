import cv2
import numpy as np
import time
import os
import threading           # ---------- 新增
import cv2                 # ---------- 新增
import psutil
from rknnlite.api import RKNNLite
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray,Float64MultiArray
from ament_index_python.packages import get_package_share_directory
import argparse

OBJ_THRESH = 0.38
NMS_THRESH = 0.2
IMG_SIZE = 640
CLASSES = ("fire")

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def process(input, mask, anchors):
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = input[..., 4]
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = input[..., 5:]

    box_xy = input[..., :2]*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = pow(input[..., 2:4]*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def save_best_five(frames):
    if not frames:
        return
    idx_max = max(range(len(frames)), key=lambda i: frames[i][1])
    for i in range(idx_max, min(idx_max + 5, len(frames))):
        img, conf = frames[i]
        fname = f'best_{datetime.now():%Y%m%d_%H%M%S}_{i-idx_max+1}_{conf:.2f}.jpg'
        cv2.imwrite(fname, img)
        print(f'[SAVE] {fname}')

# ================= 改造后的 uvc_preview ================
def uvc_preview(
                flight_flag: threading.Event,
                stop_event,
                device_index,
                window_name,
                image_pub,
                clock,
                rknn_lite,        # 新增
                img_size):        # 新增
    
    """仅负责采集 / 翻转 / 发布，不再创建 Node。"""
    bridge = CvBridge()
    frames  = []
    cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f'❌ 无法打开摄像头 {device_index}')
        stop_event.clear();  return

    
    pub_period  = 0.1
    last_pub    = 0.0
    in_flie = False           # 当前是否正在飞行
    pre_conf=0
    while stop_event.is_set():
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        # ---------- 1) 上下翻转 ----------
        frame_bgr = cv2.flip(frame_bgr, 0)

        # ---------- 2) 推理前预处理 ----------
        img_rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size))
        img_input   = np.expand_dims(img_resized, 0)

        # ---------- 3) RKNN 推理 ----------
        outputs = rknn_lite.inference(inputs=[img_input], data_format=['nhwc'])

        # ---------- 4) 后处理 ----------
        # 4.1 reshape / transpose（与您原有代码一致）
        input0 = outputs[0].reshape([3, -1]+list(outputs[0].shape[-2:]))
        input1 = outputs[1].reshape([3, -1]+list(outputs[1].shape[-2:]))
        input2 = outputs[2].reshape([3, -1]+list(outputs[2].shape[-2:]))
        input_t = [np.transpose(x, (2, 3, 0, 1)) for x in (input0, input1, input2)]

        boxes, classes, scores = yolov5_post_process(input_t)

                # ---------- 空结果防护 ----------
        if boxes is None or len(boxes) == 0:
            # 本帧没有检测到目标：仅发布翻转图像，不画框、不写置信度
            cv2.imshow(window_name, frame_bgr)
            if cv2.waitKey(1) & 0xFF == 27:
                stop_event.clear()
            # 10 Hz 发布仍按原频率进行
            now = time.time()
            if now - last_pub >= pub_period:
                msg = bridge.cv2_to_imgmsg(frame_bgr, encoding='bgr8')
                msg.header.stamp = clock.now().to_msg()
                image_pub.publish(msg)
                last_pub = now
            continue                      # 跳过后续绘框代码


        # ---------- 5) 可视化 ----------
        for (x1, y1, x2, y2), cls, conf in zip(boxes, classes, scores):
            # 坐标是缩放后分辨率下的，需要还原到原图
            x_scale = frame_bgr.shape[1] / img_size
            y_scale = frame_bgr.shape[0] / img_size
            p1 = (int(x1 * x_scale), int(y1 * y_scale))
            p2 = (int(x2 * x_scale), int(y2 * y_scale))
            cv2.rectangle(frame_bgr, p1, p2, (0, 255, 0), 2)
            label = f'{conf:.2f}'
            cv2.putText(frame_bgr, label, (p1[0], max(p1[1]-5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if len(scores):                                  # 本帧至少检测到 1 个目标
            best_conf = float(scores[0]) 
            
                                # scores 已按置信度降序
            text = f'conf: {best_conf:.2f}'
            # 右上角坐标（距右边缘 120 px，距上边缘 25 px）
            cv2.putText(frame_bgr,
                        text,
                        (frame_bgr.shape[1] - 120, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,               # 字体大小
                        (255, 255, 0),     # BGR 颜色：黄色
                        2,                 # 线宽
                        cv2.LINE_AA)
        if flight_flag.is_set():
                in_flie=True
                if best_conf>pre_conf:
                    frames.clear()
                    frames.append(frame_bgr.copy())
                    pre_conf=best_conf
                elif frames.count<5:
                    frames.append(frame_bgr.copy())
        elif in_flie:
            save_best_five(frames)         # 保存最高帧+后4帧
            frames.clear()
            in_flie = False
        # ---------- 6) 实时显示 ----------
        cv2.imshow(window_name, frame_bgr)
        if cv2.waitKey(1) & 0xFF == 27:
            stop_event.clear(); break
        
        # ---------- 7) 10 Hz 发布 ----------
        now = time.time()
        if now - last_pub >= pub_period:
            msg = bridge.cv2_to_imgmsg(frame_bgr, encoding='bgr8')
            msg.header.stamp = clock.now().to_msg()
            image_pub.publish(msg)
            last_pub = now

    cap.release(); cv2.destroyWindow(window_name)


def filter_boxes(boxes, box_confidences, box_class_probs):
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

class FireDetectionNode(Node):
    def __init__(self):
        super().__init__('fire_detection_node')

        
        # 初始化RKNN模型
        self.rknnModel = os.path.join(get_package_share_directory('move_to_gps_target_real_fly'), "rknnModel/fire.rknn")
        self.rknn_lite = RKNNLite()
        
        # 加载RKNN模型
        ret = self.rknn_lite.load_rknn(self.rknnModel)
        if ret != 0:
            self.get_logger().error("Load RKNN Model failed")
            exit(ret)
        
        # 初始化运行时环境
        ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            self.get_logger().error("Init runtime environment failed")
            exit(ret)
        
        # 初始化CV Bridge
        self.bridge = CvBridge()
        
        # 创建图像订阅
        self.subscription = self.create_subscription(
            Image,
            '/camera',
            self.image_callback,
            10  # QoS profile depth
        )
        
        # 创建火焰检测结果发布
        self.detection_pub = self.create_publisher(Float64MultiArray, "/fire_yolo_detection", 10)

        # 1) 创建共享对象
        self.flight_flag = threading.Event() 
        self.mode_sub = self.create_subscription(
            String, '/apm_drone/current_mode_state',
            self.mode_cb, 10)

        self.img_size = 640               # 自己定义
        self.stop_event = threading.Event(); self.stop_event.set()
        image_pub = self.create_publisher(Image, '/camera', 10)
        clock = self.get_clock()                 # <- 这里返回 Clock 实例
        self.preview_thread = threading.Thread(
            target=uvc_preview,
            args=(
                self.flight_flag,
                self.stop_event,
                  "/dev/video0",
                  'UVC Preview',
                  image_pub,        # ← 现在已存在
                  clock,
                  self.rknn_lite,
                  self.img_size),
            daemon=True)
        self.preview_thread.start()

    def mode_cb(self, msg: String):
        mode = msg.data.lower()
        if mode == 'armed':
            self.flight_flag.set()          # 进入飞行
        elif mode == 'disarmed':
            self.flight_flag.clear()        # 飞行结束（线程收到后会触发保存）
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {str(e)}")
            return

        # 图像预处理
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        
        # 模型推理
        img_input = np.expand_dims(img_resized, 0)
        outputs = self.rknn_lite.inference(inputs=[img_input], data_format=['nhwc'])
        
        # 后处理
        input0_data = outputs[0].reshape([3, -1]+list(outputs[0].shape[-2:]))
        input1_data = outputs[1].reshape([3, -1]+list(outputs[1].shape[-2:]))
        input2_data = outputs[2].reshape([3, -1]+list(outputs[2].shape[-2:]))
        
        input_data = [
            np.transpose(input0_data, (2, 3, 0, 1)),
            np.transpose(input1_data, (2, 3, 0, 1)),
            np.transpose(input2_data, (2, 3, 0, 1))
        ]
        
        boxes, classes, scores = yolov5_post_process(input_data)

        # 处理检测结果
        if boxes is not None and len(boxes) > 0:
            # 获取原始时间戳（使用双精度保持精度）
            stamp_sec = np.float64(msg.header.stamp.sec)
            stamp_nsec = np.float64(msg.header.stamp.nanosec)
            
            # 为每个检测框创建独立消息
            if boxes is not None and len(boxes) > 0:
                # 获取原始时间戳（使用Python原生类型）
                stamp_sec = float(msg.header.stamp.sec)
                stamp_nsec = float(msg.header.stamp.nanosec)
                
                # 为每个检测框创建独立消息
                for i in range(len(boxes)):
                    box = boxes[i]
                    score = float(scores[i])
                    
                    # 创建消息容器（确保消息类型匹配）
                    detection_msg = Float64MultiArray()
                    
                    # 转换所有值为Python原生float
                    detection_msg.data = [
                        float(box[0]),   # x1
                        float(box[1]),   # y1
                        float(box[2]),   # x2
                        float(box[3]),   # y2
                        score,           # 置信度
                        stamp_sec,       # 秒部分
                        stamp_nsec       # 纳秒部分
                    ]
                    
                    # 发布独立消息
                    self.detection_pub.publish(detection_msg)

    def __del__(self):
        self.rknn_lite.release()
        cv2.destroyAllWindows()
        self.get_logger().info("Fire detection node shutdown cleanly")

def main(args=None):
    parser = argparse.ArgumentParser(description="Fire Detection Node")
    parser.add_argument('--cpu', type=int, default=None, help="CPU core to bind to")
    parser.add_argument('--cam', type=int, default=0,
                        help="UVC 摄像头设备索引，默认 /dev/video0")
    parsed_args, unknown_args = parser.parse_known_args(args)


    stop_event = threading.Event()
    stop_event.set()                                       # 线程运行标志
  


    rclpy.init(args=unknown_args)

    node = FireDetectionNode()
   #   node  = rclpy.create_node('fire_detection_node')
    # image_pub = node.create_publisher(Image, '/camera', 10)
    # clock = node.get_clock()                 # <- 这里返回 Clock 实例

    # preview_thread = threading.Thread(
    #     target=uvc_preview,
    #     args=(stop_event,                     # ① 退出事件
    #         parsed_args.cam,                # ② 摄像头索引
    #         'UVC Preview',                  # ③ 窗口名
    #         image_pub,                      # ④ 发布器
    #         clock),                         # ⑤ Clock 句柄  ★务必传入
    #     daemon=True)
    # preview_thread.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()