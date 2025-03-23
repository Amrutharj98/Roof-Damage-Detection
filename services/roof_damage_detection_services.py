import numpy as np
from tensorflow.keras.models import Model, load_model
import cv2
import json
import os
from ultralytics import YOLO
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

tile_identification_model = load_model('TileClassificationEnsemble.h5')
roof_detection_model = load_model('RoofIdentificationAutoEncoderModel1.h5')
roof_segmentation_model = load_model('roof_segmentation_model2.h5')
damage_detection_model = YOLO('runs/detect/roof_damage_yolo/weights/best.pt')
class RoofDamageDetectionServices:
    def tile_type_identification(imagE, segmented_image):
        "Function to identify the tile type of the roof"
        image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        # Resize and preprocess image
        image_resized = cv2.resize(image, (224, 224))  # Resize to (224, 224)
        image_normalized = image_resized / 255  # Normalize image to [0, 1]
        image_input = np.expand_dims(image_normalized, axis=0)
        predictions = tile_identification_model.predict(image_input)
        predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get predicted class index
        class_names=["Clay Roof Tile","Natural Slate Roof Tile"]
        predicted_class_name = class_names[predicted_class_index]
        res = RoofDamageDetectionServices.damage_detection_and_level_identification(imagE)
        res.insert(0,predicted_class_name)
        return res

    def roof_segmentation(image):
        "Function to segment the roof"
        target_size=(256, 256)
        image_resized = cv2.resize(image, target_size)
        image_normalized = image_resized / 255.0
        processed_image = np.expand_dims(image_normalized, axis=0)
        predicted_mask = roof_segmentation_model.predict(processed_image)[0, :, :, 0]
        mask_resized = cv2.resize(predicted_mask, (image.shape[1], image.shape[0]))
        mask_resized = (mask_resized > 0.01).astype(np.uint8)  # Convert to binary mask
        segmented_image = cv2.bitwise_and(image, image, mask=mask_resized)
        # Save the segmented image
        save_path="E:/Roof Damage Detection/segmented_images"
        image_path = os.path.join(save_path, "segmented_image.jpg")
        cv2.imwrite(image_path, segmented_image)
        res = RoofDamageDetectionServices.tile_type_identification(image, segmented_image)
        res.insert(0,image_path)
        return res
    def calculate_union_area(boxes):
        shapely_boxes = [shapely_box(x1, y1, x2, y2) for x1, y1, x2, y2 in boxes]
        if not shapely_boxes:
            return 0
        union = unary_union(shapely_boxes)
        return union.area

    def damage_detection_and_level_identification(img):
        "Function to identify damaged regions and "
        results = damage_detection_model.predict(
            source=img,  
            imgsz=640,    
            conf=0.25,    
            save=False    
        )
        for result in results:
            img_height, img_width, _ = img.shape 

            total_damage_area = 0
            image_area = img_width * img_height

            # Check if there are bounding boxes
            if not result.boxes:
                save_path="E:/Roof Damage Detection/damaged_images"
                img_path = os.path.join(save_path, "damaged_image.jpg")
                cv2.imwrite(img_path, img)
                res = [img_path, 0]
                continue
        # Extract bounding box coordinates and areas
            boxes = [
                tuple(map(float, box)) for box in result.boxes.xyxy
            ]  # Convert to (x1, y1, x2, y2) format
            areas = [
                max(0, (x2 - x1)) * max(0, (y2 - y1)) for x1, y1, x2, y2 in boxes
            ]

            # Sort boxes by area in descending order
            sorted_boxes = sorted(zip(boxes, areas), key=lambda x: x[1], reverse=True)

            # Keep only the top 3 regions
            top_boxes = sorted_boxes[:3]

            # Calculate the total area of the top 3 regions
            total_damage_area = sum(area for _, area in top_boxes)

            # Calculate damage percentage
            damage_percentage = (total_damage_area / image_area) * 100 if image_area > 0 else 0
            # Draw only the top 3 bounding boxes
            for box, area in top_boxes:
                x1, y1, x2, y2 = box
                color = (0, 255, 0)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            # Display the result
            save_path="E:/Roof Damage Detection/damaged_images"
            img_path = os.path.join(save_path, "damaged_image.jpg")
            cv2.imwrite(img_path, img)
            if damage_percentage > 0:
                res = [img_path, damage_percentage]
            else:
                save_path="E:/Roof Damage Detection/damaged_images"
                img_path = os.path.join(save_path, "damaged_image.jpg")
                cv2.imwrite(img_path, img)
                res = [img_path, 0]
        return res       


    def roof_or_not_detection(image):
        "Function to identify roof and non-roof images"
        image_resized = cv2.resize(image, (224, 224))  
        image_normalized = image_resized / 255  
        image_input = np.expand_dims(image_normalized, axis=0)
        predictions = roof_detection_model.predict(image_input)
        # Compute reconstruction error for each test image
        reconstruction_errors = np.mean(np.square(image_input - predictions), axis=(1, 2, 3))
        if reconstruction_errors > 0.08:
            result = {
                "roof":"No Roof Detected"
            }
        else:
            res = RoofDamageDetectionServices.roof_segmentation(image)
            result = {
                "roof":"Roof Detected",
                "segmented_image":res[0],
                "tile_type":res[1],
                "damage_detected_regions":res[2],
                "damage_percentage":res[3]
            }
        return result
