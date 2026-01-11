import cv2
import numpy as np
import time
from ultralytics import YOLO

from settings import (YOLO_MODEL_PATH, FEATURE_MODEL_PATH, OBJECT_CLASSES,
                     COSINE_DISTANCE_THRESHOLD, FEATURE_BUDGET, IOU_THRESHOLD,
                     MAX_TRACK_AGE, TRACK_CONFIRMATION_HITS)
from detection import ObjectDetection, create_feature_encoder
from distance import AppearanceMetric
from track import MultiObjectTracker
from visualization import draw_tracking_boxes


def process_video(input_video_path, output_video_path):
    print(f"Starting tracking process: {input_video_path}")

    feature_encoder = create_feature_encoder(FEATURE_MODEL_PATH, batch_size=1)
    appearance_metric = AppearanceMetric("cosine", COSINE_DISTANCE_THRESHOLD, FEATURE_BUDGET)
    object_tracker = MultiObjectTracker(appearance_metric, IOU_THRESHOLD, MAX_TRACK_AGE, TRACK_CONFIRMATION_HITS)
    
    detection_model = YOLO(YOLO_MODEL_PATH)

    video_capture = cv2.VideoCapture(input_video_path)
    if not video_capture.isOpened():
        raise IOError(f"Failed to open video: {input_video_path}")

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    
    print(f"Video properties: {frame_width}x{frame_height}, {video_fps} FPS")

    if video_fps == 0:
        video_fps = 25
    if frame_width == 0 or frame_height == 0:
        raise ValueError("Invalid video dimensions detected")

    output_codec = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path, output_codec, video_fps, (frame_width, frame_height))
    print(f"Output will be saved to: {output_video_path}")

    detection_times = []
    total_processing_times = []
    current_frame = 0

    class_keys = list(OBJECT_CLASSES.keys())
    class_values = list(OBJECT_CLASSES.values())

    while True:
        frame_read_success, video_frame = video_capture.read()
        if not frame_read_success or video_frame is None:
            print("Video processing complete")
            break
        
        current_frame += 1
        rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)

        detection_start = time.time()
        detection_results = detection_model.predict(rgb_frame, imgsz=640, conf=0.5, verbose=False)
        detection_end = time.time()

        frame_detections = detection_results[0].boxes

        if current_frame <= 5:
            print(f"\nFrame {current_frame}: {len(frame_detections)} objects detected")
            for detection_idx, detected_box in enumerate(frame_detections):
                detected_class = int(detected_box.cls[0])
                detection_confidence = float(detected_box.conf[0])
                class_name = detection_model.names[detected_class]
                print(f"  Object {detection_idx}: {class_name} (confidence={detection_confidence:.2f})")

        if frame_detections is None or len(frame_detections) == 0:
            video_writer.write(video_frame)
            continue

        detected_boxes = []
        confidence_scores = []
        class_names = []
        
        for detected_box in frame_detections:
            box_coords = detected_box.xyxy[0].cpu().numpy()
            detected_class = int(detected_box.cls[0])
            detection_confidence = float(detected_box.conf[0])
            class_name = detection_model.names[detected_class]
            standardized_name = class_name.replace(' ', '-')

            x_min, y_min, x_max, y_max = box_coords
            detected_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
            confidence_scores.append(detection_confidence)
            class_names.append(standardized_name)

        detected_boxes = np.array(detected_boxes)
        confidence_scores = np.array(confidence_scores)
        class_names = np.array(class_names)

        appearance_features = np.array(feature_encoder(rgb_frame, detected_boxes))
        
        detection_objects = [
            ObjectDetection(box, score, name, feature)
            for box, score, name, feature in zip(detected_boxes, confidence_scores, 
                                                 class_names, appearance_features)
        ]

        object_tracker.predict_all_tracks()
        object_tracker.update_with_detections(detection_objects)

        tracked_boxes = []
        for tracked_object in object_tracker.active_tracks:
            if not tracked_object.is_confirmed() or tracked_object.frames_since_update > 5:
                continue
            
            object_box = tracked_object.to_top_left_bottom_right()
            object_class = tracked_object.get_object_category()
            object_id = tracked_object.unique_id
            
            class_index = class_keys[class_values.index(object_class)] if object_class in class_values else 0
            tracked_boxes.append(object_box.tolist() + [object_id, class_index])

        annotated_frame = draw_tracking_boxes(rgb_frame, tracked_boxes, CLASSES=OBJECT_CLASSES, tracking=True)

        processing_end = time.time()
        detection_times.append(detection_end - detection_start)
        total_processing_times.append(processing_end - detection_start)
        
        detection_times = detection_times[-20:]
        total_processing_times = total_processing_times[-20:]
        
        avg_detection_time = sum(detection_times) / len(detection_times)
        avg_total_time = sum(total_processing_times) / len(total_processing_times)
        
        detection_fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0
        total_fps = 1.0 / avg_total_time if avg_total_time > 0 else 0
        
        cv2.putText(annotated_frame, f"FPS: {detection_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)

        if current_frame % 20 == 0:
            print(f"Processed {current_frame} frames, Detection FPS={detection_fps:.1f}, Total FPS={total_fps:.1f}")

    video_capture.release()
    video_writer.release()
    print(f"Tracking complete. Output saved: {output_video_path}")


if __name__ == "__main__":
    input_path = "./Football match.mp4"
    output_path = "./Football_match_tracked.mp4"
    process_video(input_path, output_path)
