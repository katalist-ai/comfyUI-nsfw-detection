import math
import os

import cv2
import numpy as np
import onnxruntime
import torch

__labels = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

classes_to_detect = [
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
]


def preprocess(img: np.ndarray, target_size=320):
    # assume image of shape (H, W, C), RGB, float32
    img_height, img_width = img.shape[:2]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    aspect = img_width / img_height

    if img_height > img_width:
        new_height = target_size
        new_width = int(round(target_size * aspect))
    else:
        new_width = target_size
        new_height = int(round(target_size / aspect))

    resize_factor = math.sqrt(
        (img_width ** 2 + img_height ** 2) / (new_width ** 2 + new_height ** 2)
    )

    img = cv2.resize(img, (new_width, new_height))

    pad_x = target_size - new_width
    pad_y = target_size - new_height

    pad_top, pad_bottom = [int(i) for i in np.floor([pad_y, pad_y]) / 2]
    pad_left, pad_right = [int(i) for i in np.floor([pad_x, pad_x]) / 2]

    img = cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    img = cv2.resize(img, (target_size, target_size))

    image_data = img.astype("float32")
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)
    return image_data, resize_factor, pad_left, pad_top


def _postprocess(output, resize_factor, pad_left, pad_top):
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)

        if max_score >= 0.2:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int(round((x - w * 0.5 - pad_left) * resize_factor))
            top = int(round((y - h * 0.5 - pad_top) * resize_factor))
            width = int(round(w * resize_factor))
            height = int(round(h * resize_factor))
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)

    detections = []
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        detections.append(
            {"class": __labels[class_id], "score": float(score), "box": box}
        )

    return detections


class NudenetDetector:

    def __init__(self, providers=None):
        self.onnx_session = onnxruntime.InferenceSession(
            os.path.join(os.path.dirname(__file__), "best.onnx"),
            providers=["CPUExecutionProvider"]
        )
        model_inputs = self.onnx_session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]  # 320
        self.input_height = input_shape[3]  # 320
        self.input_name = model_inputs[0].name

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("IMAGE",)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)

    FUNCTION = "detect_and_blur"

    # OUTPUT_NODE = False

    CATEGORY = "nsfw"

    def detect_and_blur(self, image: torch.Tensor):
        print(os.path.exists(os.path.join(os.path.dirname(__file__), "best.onnx")))
        print('detect and blur NEW')
        all_detections = []
        all_imgs = []
        print('image sizes:', image.size())
        for i in range(len(image)):
            img = image[i].numpy()
            print(img.dtype)
            preprocessed_image, resize_factor, pad_left, pad_top = preprocess(
                img, self.input_width
            )
            print(preprocessed_image.dtype, preprocessed_image.min(), preprocessed_image.max())
            outputs = self.onnx_session.run(None, {self.input_name: preprocessed_image})
            print(outputs[0].shape)
            detections = _postprocess(outputs, resize_factor, pad_left, pad_top)
            print('detections: ', detections)
            detections = [
                detection for detection in detections if detection["class"] in classes_to_detect
            ]
            all_detections.append(detections)
            for detection in detections:
                print('masking: ', detection)
                box = detection["box"]
                x, y, w, h = box[0], box[1], box[2], box[3]
                # change these pixels to pure black
                img[y: y + h, x: x + w] = (0, 0, 0)
            all_imgs.append(img)
        return (torch.tensor(np.array(all_imgs)),)

    # @classmethod
    # def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "NudenetDetector": NudenetDetector
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "NudenetDetector": "NudenetDetector"
}
