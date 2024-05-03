import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import math
import os
from deep_sort.deep_sort import DeepSort
from sklearn.cluster import DBSCAN

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class VideoCameraYolo:
    def __init__(self):
        self.model_yolo = YOLO('yolov8l.yaml')
        self.mp_pose = mp.solutions.pose
        self.middle_position = (320, 240)
        self.pose = self.mp_pose.Pose()
        self.unique_track_ids = set()
        self.frames = []
        self.angle_left = ""
        self.angle_right = ""
        self.boundary_polygon = np.array([[20, 346], [280, 140], [580, 165], [760, 380]], np.int32)
        self.deep_sort_wights = 'deep_sort/deep/checkpoint/ckpt.t7'
        self.tracker = DeepSort(model_path=self.deep_sort_wights, max_age=10, min_confidence=0.5)
        self.throwing_persons = {}
        self.throwing_frames_threshold = 5
        self.throwing_frames_counter = {}

        self.continous_id = 0

        self.eps = 50  # DBSCAN epsilon parameter
        self.min_samples = 2  # DBSCAN min_samples parameter
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)

    def calculate_elbow_rotation(self, shoulder, elbow):
        direction_vector = elbow - shoulder
        return direction_vector

    def hand_mesaure(self, direction_vector_left, direction_vector_right, img, left_elbow_point, right_elbow_point,
                     track_id):
        distance = 3
        if direction_vector_left[1] < 0:
            self.angle_left = "left hand up"
        else:
            self.angle_left = "left hand down"

        if direction_vector_right[1] < 0:
            self.angle_right = "right hand up"
        else:
            self.angle_right = "right hand down"

        # Combine both angles to determine if both hands are up, down, or mixed
        if self.angle_left == "left hand up" and self.angle_right == "right hand up":
            angle = "both hands up"
        elif self.angle_left == "left hand down" and self.angle_right == "right hand down":
            angle = "both hands down"
        else:
            angle = "mixed"

        if distance < 1500:
            if angle == "both hands up" or angle == 'mixed':
                left_hand_moving_towards_middle = left_elbow_point[0] > self.middle_position[0]
                right_hand_moving_towards_middle = right_elbow_point[0] < self.middle_position[0]

                self.throwing_frames_counter[track_id] = self.throwing_frames_counter.get(track_id, 0) + 1

                # Check if throwing frames threshold reached
                if self.throwing_frames_counter[track_id] >= self.throwing_frames_threshold:
                    # Only update throwing_persons dictionary if throwing action detected for threshold frames
                    self.throwing_persons[track_id] = True
                    cv2.putText(img, f"Throwing something!", (img.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    return track_id

        return None  # Return None if throwing action not detected

    def check_intersection(self, bounding_box):
        for corner in bounding_box:
            # Extract x and y coordinates from corner and ensure they are integers
            x, y = int(corner[0]), int(corner[1])
            if not isinstance(x, int) or not isinstance(y, int):
                print(f"Error: Invalid data type for coordinates: ({x}, {y})")
                return False

            # Check if the point is inside the boundary polygon
            if cv2.pointPolygonTest(self.boundary_polygon, (x, y), False) >= 0:
                return True
        return False

    def gen_frames(self):
        video_path = 'E:/offical_porojects/wevois/people_counting_project/trail2.mp4'
        if not os.path.exists(video_path):
            print("Error: Video file does not exist")
        cap = cv2.VideoCapture(video_path)
        self.model = YOLO("yolo-Weights/yolov8l.pt")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output2.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            try:
                success, img = cap.read()
                if not success:
                    print("Can't receive frame (stream end?). Exiting ...")
            except Exception as e:
                print("Error occurred during video processing:", e)
                break
            cv2.polylines(img, [self.boundary_polygon], isClosed=True, color=(0, 255, 0), thickness=2)
            results_yolo = self.model(img, stream=True, classes=0, conf=0.7)
            angle = 0
            class_names = ['person']
            for r in results_yolo:
                boxes = r.boxes

                probs = r.probs
                cls = boxes.cls.tolist()
                xyxy = boxes.xyxy
                conf = boxes.conf
                xywh = boxes.xywh
                for class_index in cls:
                    class_name = class_names[int(class_index)]
            pred_cls = np.array(cls)
            conf = conf.detach().cpu().numpy()
            xyxy = xyxy.detach().cpu().numpy()
            bboxes_xywh = xywh
            bboxes_xywh = xywh.cpu().numpy()
            bboxes_xywh = np.array(bboxes_xywh, dtype=float)

            try:
                tracks = self.tracker.update(bboxes_xywh, conf, img)
            except:
                continue

            for track in self.tracker.tracker.tracks:
                try:
                    track_id = track.track_id
                    hits = track.hits
                    x1, y1, x2, y2 = track.to_tlbr()
                    w = x2 - x1
                    h = y2 - y1

                    red_color = (0, 0, 255)
                    blue_color = (255, 0, 0)
                    green_color = (0, 255, 0)

                    color_id = track_id % 3
                    if color_id == 0:
                        color = red_color
                    elif color_id == 1:
                        color = blue_color
                    else:
                        color = green_color

                    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    person_roi = img[y1:y2, x1:x2]
                    bounding_box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


                    if self.check_intersection(bounding_box):
                        if True:
                            person_results = self.pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))

                            try:
                                left_shoulder_landmark = person_results.pose_landmarks.landmark[
                                    mp_pose.PoseLandmark.LEFT_SHOULDER]
                                left_elbow_landmark = person_results.pose_landmarks.landmark[
                                    mp_pose.PoseLandmark.LEFT_ELBOW]

                                Right_shoulder_landmark = person_results.pose_landmarks.landmark[
                                    mp_pose.PoseLandmark.RIGHT_SHOULDER]
                                Right_elbow_landmark = person_results.pose_landmarks.landmark[
                                    mp_pose.PoseLandmark.RIGHT_ELBOW]

                                # Convert landmark coordinates to numpy arrays
                                left_shoulder_point = np.array(
                                    [left_shoulder_landmark.x * img.shape[1],
                                     left_shoulder_landmark.y * img.shape[0],
                                     left_shoulder_landmark.z * img.shape[
                                         1]])  # Z coordinate is included for 3D pose
                                left_elbow_point = np.array(
                                    [left_elbow_landmark.x * img.shape[1], left_elbow_landmark.y * img.shape[0],
                                     left_elbow_landmark.z * img.shape[1]])

                                right_shoulder_point = np.array(
                                    [Right_shoulder_landmark.x * img.shape[1],
                                     Right_shoulder_landmark.y * img.shape[0],
                                     Right_shoulder_landmark.z * img.shape[
                                         1]])  # Z coordinate is included for 3D pose
                                right_elbow_point = np.array(
                                    [Right_elbow_landmark.x * img.shape[1], Right_elbow_landmark.y * img.shape[0],
                                     Right_elbow_landmark.z * img.shape[1]])

                                direction_vector_right = self.calculate_elbow_rotation(right_shoulder_point,
                                                                                       right_elbow_point)
                                direction_vector_left = self.calculate_elbow_rotation(left_shoulder_point,
                                                                                      left_elbow_point)
                                angle = self.hand_mesaure(direction_vector_left, direction_vector_right, img,
                                                          left_elbow_point, right_elbow_point, track_id)

                            except:
                                pass

                    text_color = (0, 0, 0)
                    cv2.putText(img, f"{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (128, 0, 128), 2, cv2.LINE_AA)




                except:
                    pass

                else:
                    pass

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                color = (123, 233, 0)
                thickness = 2
            cv2.putText(img, f"{self.throwing_persons}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 122), 1, cv2.LINE_AA)

            out.write(img)

            try:
                _, jpeg = cv2.imencode('.jpg', img)
                frame_bytes = jpeg.tobytes()
                yield img
            except:
                pass

        cap.release()
        out.release()


camera = VideoCameraYolo()

for frame in camera.gen_frames():
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
