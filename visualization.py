import cv2
import numpy as np


def draw_tracking_boxes(source_image, tracked_objects, class_mapping, 
                        show_labels=True, show_scores=True, text_color=(255, 255, 255)):
    person_box_color = (0, 0, 255)
    ball_box_color = (0, 255, 0)
    default_box_color = (255, 0, 0)

    image_height, image_width = source_image.shape[:2]

    for detection_data in tracked_objects:
        box_coordinates = np.array(detection_data[:4], dtype=np.int32)
        confidence_value = float(detection_data[4]) if len(detection_data) > 4 else 0.0
        class_index = int(detection_data[5]) if len(detection_data) > 5 else 0
        
        class_label = str(class_mapping[class_index]).lower()
        normalized_label = class_label.replace('_', ' ').replace('-', ' ').strip()

        if 'person' in normalized_label:
            box_color = person_box_color
        elif 'ball' in normalized_label:
            box_color = ball_box_color
        else:
            box_color = default_box_color

        line_thickness = max(1, int(0.6 * (image_height + image_width) / 1000))
        text_scale = 0.75 * line_thickness
        
        x1, y1, x2, y2 = int(box_coordinates[0]), int(box_coordinates[1]), \
                         int(box_coordinates[2]), int(box_coordinates[3])

        cv2.rectangle(source_image, (x1, y1), (x2, y2), box_color, line_thickness * 2)

        if show_labels:
            score_text = f" {confidence_value:.2f}" if show_scores else ""
            display_label = f"{class_label}{score_text}"
            
            (label_width, label_height), baseline = cv2.getTextSize(
                display_label, cv2.FONT_HERSHEY_COMPLEX_SMALL, text_scale, thickness=line_thickness)
            
            label_y_position = max(0, y1 - label_height - baseline)
            
            cv2.rectangle(source_image, (x1, label_y_position),
                         (x1 + label_width, y1), box_color, thickness=cv2.FILLED)
            
            cv2.putText(source_image, display_label, (x1, y1 - 4),
                       cv2.FONT_HERSHEY_COMPLEX_SMALL, text_scale, text_color, 
                       line_thickness, lineType=cv2.LINE_AA)

    return source_image
