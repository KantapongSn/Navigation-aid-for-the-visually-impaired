import cv2
import numpy as np
import pyrealsense2 as rs
import pyttsx3
import threading
from image_processing import adjust_brightness_contrast, preprocess_image
from sound_utils import blink_sound
from yolo_utils import load_yolo_model
from canny import find_contours

# โหลด YOLO โมเดล
net, LABELS, COLORS = load_yolo_model()

# กำหนด RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# กำหนดค่าพื้นฐาน
fs = 43200
base_frequency = 440
min_distance = 0.1
max_distance = 2.0

# กำหนดเสียงพูด
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ฟังก์ชันสำหรับการแสดงผลใน thread
def display_depth_colormap(depth_image):
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
    )
    cv2.imshow("Depth Colormap", depth_colormap)

def display_realsense_frame(adjusted_frame):
    cv2.imshow('RealSense', adjusted_frame)

try:
    while True:
        # อ่านเฟรมจาก RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        (H, W) = frame.shape[:2]

        # ปรับความสว่างและความคมชัด
        adjusted_frame = adjust_brightness_contrast(frame, alpha=1.2, beta=20)

        # การประมวลผลภาพ
        edges = preprocess_image(adjusted_frame)
        contours = find_contours(edges)

        canny_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 150:
                (x, y, w, h) = cv2.boundingRect(contour)
                canny_boxes.append([x, y, w, h])

        # รวมกล่องที่ใกล้กัน
        merged_boxes = []
        while canny_boxes:
            base_box = canny_boxes.pop(0)
            (bx, by, bw, bh) = base_box
            merged = False
            for i in range(len(canny_boxes)):
                (nx, ny, nw, nh) = canny_boxes[i]
                base_center = (bx + bw // 2, by + bh // 2)
                new_center = (nx + nw // 2, ny + nh // 2)
                distance = np.sqrt((base_center[0] - new_center[0]) ** 2 +
                                   (base_center[1] - new_center[1]) ** 2)
                depth_base = depth_frame.get_distance(bx + bw // 2, by + bh // 2)
                depth_new = depth_frame.get_distance(nx + nw // 2, ny + nh // 2)
                if distance < 150 and abs(depth_base - depth_new) < 0.5:
                    bx = min(bx, nx)
                    by = min(by, ny)
                    bw = max(bx + bw, nx + nw) - bx
                    bh = max(by + bh, ny + nh) - by
                    merged = True
                    canny_boxes.pop(i)
                    break
            if merged:
                canny_boxes.append([bx, by, bw, bh])
            else:
                merged_boxes.append([bx, by, bw, bh])

        closest_depth = float('inf')

        for (x, y, w, h) in merged_boxes:
            depth = depth_frame.get_distance(x + w // 2, y + h // 2)
            if 0 < depth < 1.5:
                color = (0, 0, 255)
                cv2.rectangle(adjusted_frame, (x, y), (x + w, y + h), color, 2)
                text = "Depth: {:.4f}m".format(depth)
                cv2.putText(adjusted_frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # คำนวณ duration ตาม depth
                if 1.5 > depth > 1.3:
                    duration = 1
                elif 1.3 >= depth > 1:
                    duration = 0.7
                elif 1 >= depth > 0.7:
                    duration = 0.5
                elif 0.7 >= depth > 0.5:
                    duration = 0.2
                elif 0.5 >= depth > 0:
                    duration = 0.1
                else:
                    duration = 0

                print(f"Contour Duration: {duration:.2f} seconds")
                blink_sound(duration, 432, fs)

        # YOLO object detection
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(adjusted_frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        stair_detected = False
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.7:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    depth = depth_frame.get_distance(x + width // 2, y + height // 2)
                    if LABELS[classID] == "Stairs" and 0 < depth < 3.0:
                        color = (0, 255, 0)
                        cv2.rectangle(adjusted_frame, (x, y), (x + width, y + height), color, 2)
                        text = "{}: {:.4f}m".format(LABELS[classID], depth)
                        cv2.putText(adjusted_frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        stair_detected = True

        if stair_detected:
            engine.say("stair")
            engine.runAndWait()
            print("Stairs!!!!")

        # วาดกรอบรอบวัตถุในภาพ depth
        depth_threshold = cv2.inRange(depth_image, 10, 1500)
        contours, _ = cv2.findContours(depth_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 150:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(adjusted_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # เรียกใช้ฟังก์ชันในการแสดงผลใน thread
        threading.Thread(target=display_depth_colormap, args=(depth_image,)).start()
        threading.Thread(target=display_realsense_frame, args=(adjusted_frame,)).start()

        # รอการกดปุ่ม 'q' เพื่อหยุด
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
'''
import cv2
import numpy as np
import pyrealsense2 as rs
import pyttsx3
import threading
from image_processing import adjust_brightness_contrast, preprocess_image
from sound_utils import blink_sound
from yolo_utils import load_yolo_model
from canny import find_contours

# โหลด YOLO โมเดล
net, LABELS, COLORS = load_yolo_model()

# กำหนด RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# กำหนดค่าพื้นฐาน
fs = 44100
base_frequency = 440
min_distance = 0.1
max_distance = 2.0

# กำหนดเสียงพูด
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ฟังก์ชันสำหรับการแสดงผลใน thread
def display_depth_colormap(depth_image):
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
    )
    cv2.imshow("Depth Colormap", depth_colormap)

def display_realsense_frame(adjusted_frame):
    cv2.imshow('RealSense', adjusted_frame)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        (H, W) = frame.shape[:2]

        # ปรับความสว่างและความคมชัด
        adjusted_frame = adjust_brightness_contrast(frame, alpha=1.2, beta=20)

        # การประมวลผลภาพ
        edges = preprocess_image(adjusted_frame)
        contours = find_contours(edges)

        canny_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 150:
                (x, y, w, h) = cv2.boundingRect(contour)
                canny_boxes.append([x, y, w, h])

        # รวมกล่องที่ใกล้กัน
        merged_boxes = []
        while canny_boxes:
            base_box = canny_boxes.pop(0)
            (bx, by, bw, bh) = base_box
            merged = False
            for i in range(len(canny_boxes)):
                (nx, ny, nw, nh) = canny_boxes[i]
                base_center = (bx + bw // 2, by + bh // 2)
                new_center = (nx + nw // 2, ny + nh // 2)
                distance = np.sqrt((base_center[0] - new_center[0]) ** 2 + (base_center[1] - new_center[1]) ** 2)
                depth_base = depth_frame.get_distance(bx + bw // 2, by + bh // 2)
                depth_new = depth_frame.get_distance(nx + nw // 2, ny + nh // 2)
                if distance < 150 and abs(depth_base - depth_new) < 0.5:
                    bx = min(bx, nx)
                    by = min(by, ny)
                    bw = max(bx + bw, nx + nw) - bx
                    bh = max(by + bh, ny + nh) - by
                    merged = True
                    canny_boxes.pop(i)
                    break
            if merged:
                canny_boxes.append([bx, by, bw, bh])
            else:
                merged_boxes.append([bx, by, bw, bh])

        for (x, y, w, h) in merged_boxes:
            depth = depth_frame.get_distance(x + w // 2, y + h // 2)
            if 0 < depth < 1.5:
                color = (0, 0, 255)
                cv2.rectangle(adjusted_frame, (x, y), (x + w, y + h), color, 2)
                text = "Depth: {:.4f}m".format(depth)
                cv2.putText(adjusted_frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # คำนวณ delay ตาม depth
                if 1.5 > depth > 1.3:
                    delay = 1.0
                elif 1.3 >= depth > 1.0:
                    delay = 0.7
                elif 1.0 >= depth > 0.7:
                    delay = 0.5
                elif 0.7 >= depth > 0.5:
                    delay = 0.2
                elif 0.5 >= depth > 0.0:
                    delay = 0.1
                else:
                    delay = 0  # Default

                print(f"Contour Delay: {delay:.2f} seconds")

                # ส่งเสียงด้วย duration คงที่ 0.1 และ delay ตาม depth
                blink_sound(0.1, 440, fs)
                threading.Event().wait(delay)  # รอเวลาหน่วงตาม delay ที่คำนวณ

        # YOLO object detection
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(adjusted_frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        stair_detected = False

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.7:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    depth = depth_frame.get_distance(x + width // 2, y + height // 2)

                    if LABELS[classID] == "Stairs" and 0 < depth < 3.0:
                        color = (0, 255, 0)  # สีเขียว
                        cv2.rectangle(adjusted_frame, (x, y), (x + width, y + height), color, 2)
                        text = "{}: {:.4f}m".format(LABELS[classID], depth)
                        cv2.putText(adjusted_frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        stair_detected = True

        if stair_detected:
            engine.say("stair")
            engine.runAndWait()
            print("Stairs!!!!")

        # แสดงภาพ
        threading.Thread(target=display_depth_colormap, args=(depth_image,)).start()
        threading.Thread(target=display_realsense_frame, args=(adjusted_frame,)).start()

        # รอการกดปุ่ม 'q' เพื่อหยุด
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
'''

