import numpy as np
import cv2
import tensorflow.compat.v1 as tf


class ObjectDetection:
    def __init__(self, bounding_box, confidence_score, object_class, appearance_feature):
        self.bounding_box = np.asarray(bounding_box, dtype=np.float64)
        self.confidence_score = float(confidence_score)
        self.object_class = object_class
        self.appearance_feature = np.asarray(appearance_feature, dtype=np.float64)

    def get_class_name(self):
        return self.object_class

    def to_top_left_bottom_right(self):
        result = self.bounding_box.copy()
        result[2:] += result[:2]
        return result

    def to_center_aspect_height(self):
        result = self.bounding_box.copy()
        result[:2] += result[2:] / 2
        result[2] /= result[3]
        return result


class FeatureExtractor:
    def __init__(self, model_path, input_tensor="images", output_tensor="features"):
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():
            graph_definition = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, "rb") as model_file:
                graph_definition.ParseFromString(model_file.read())
                tf.import_graph_def(graph_definition, name="")
        
        self.tensorflow_session = tf.compat.v1.Session(graph=self.computation_graph)
        
        try:
            self.input_tensor = self.computation_graph.get_tensor_by_name(input_tensor)
            self.output_tensor = self.computation_graph.get_tensor_by_name(output_tensor)
        except KeyError:
            all_operations = [op.name for op in self.computation_graph.get_operations()]
            self.input_tensor = self.computation_graph.get_tensor_by_name(all_operations[0] + ":0")
            self.output_tensor = self.computation_graph.get_tensor_by_name(all_operations[-1] + ":0")
        
        self.feature_dimensions = self.output_tensor.shape.as_list()[-1]
        self.input_shape = self.input_tensor.shape.as_list()[1:]

    def __call__(self, image_data, processing_batch_size=32):
        output_features = np.zeros((len(image_data), self.feature_dimensions), np.float64)
        
        def process_in_batches(processing_function, input_dictionary, output_array, batch_size):
            total_samples = len(input_dictionary[list(input_dictionary.keys())[0]])
            for batch_start in range(0, total_samples, batch_size):
                batch_end = min(batch_start + batch_size, total_samples)
                current_batch = {key: value[batch_start:batch_end] for key, value in input_dictionary.items()}
                output_array[batch_start:batch_end] = processing_function(current_batch)
        
        process_in_batches(
            lambda batch: self.tensorflow_session.run(self.output_tensor, feed_dict=batch),
            {self.input_tensor: image_data},
            output_features,
            processing_batch_size
        )
        return output_features


def extract_image_region(source_image, box_coordinates, target_shape):
    box = np.array(box_coordinates)
    if target_shape is not None:
        desired_aspect_ratio = float(target_shape[1]) / target_shape[0]
        adjusted_width = desired_aspect_ratio * box[3]
        box[0] -= (adjusted_width - box[2]) / 2
        box[2] = adjusted_width
    
    box[2:] += box[:2]
    box = box.astype(np.int32)
    box[:2] = np.maximum(0, box[:2])
    box[2:] = np.minimum(np.asarray(source_image.shape[:2][::-1]) - 1, box[2:])
    
    if np.any(box[:2] >= box[2:]):
        return None
    
    x_start, y_start, x_end, y_end = box
    cropped_region = source_image[y_start:y_end, x_start:x_end]
    resized_region = cv2.resize(cropped_region, tuple(target_shape[::-1]))
    return resized_region


def create_feature_encoder(model_path, input_name="images:0", output_name="features:0", batch_size=32):
    feature_extractor = FeatureExtractor(model_path, input_name, output_name)
    expected_shape = feature_extractor.input_shape
    
    def encode_boxes(source_image, detection_boxes):
        extracted_patches = []
        for single_box in detection_boxes:
            image_patch = extract_image_region(source_image, single_box, expected_shape[:2])
            if image_patch is None:
                image_patch = np.random.uniform(0., 255., expected_shape).astype(np.uint8)
            extracted_patches.append(image_patch)
        all_patches = np.asarray(extracted_patches)
        return feature_extractor(all_patches, batch_size)
    
    return encode_boxes
