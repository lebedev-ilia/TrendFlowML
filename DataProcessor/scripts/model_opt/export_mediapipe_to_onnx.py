#!/usr/bin/env python3
"""
Создание Python backend модели для MediaPipe в Triton.
MediaPipe не может быть экспортирован в ONNX из-за использования Python кода,
поэтому используем Python backend для Triton напрямую.
"""

import os
import sys
import argparse
import shutil
import numpy as np
import mediapipe as mp

# Константы из core_face_landmarks
POSE_LANDMARKS = 33
POSE_DIMS = 4  # x, y, z, visibility
HAND_LANDMARKS = 21
HAND_DIMS = 3  # x, y, z
FACE_LANDMARKS = 468
FACE_DIMS = 3  # x, y, z
MAX_HANDS = 2
MAX_FACES = 1

# Базовое разрешение для пайплайна (можно сделать конфигурируемым)
H, W = 256, 256


def create_triton_python_model(
    output_path: str,
    use_pose: bool = True,
    use_hands: bool = True,
    use_face: bool = True,
):
    """Создает Python backend модель для Triton."""
    model_code = f'''"""
Triton Python backend для MediaPipe landmarks.
Использует максимальные параметры качества:
- pose: model_complexity=2, enable_segmentation=True
- hands: model_complexity=1, max_num_hands={MAX_HANDS}
- face: refine_landmarks=True, max_num_faces={MAX_FACES}
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils
import mediapipe as mp

POSE_LANDMARKS = {POSE_LANDMARKS}
POSE_DIMS = {POSE_DIMS}
HAND_LANDMARKS = {HAND_LANDMARKS}
HAND_DIMS = {HAND_DIMS}
FACE_LANDMARKS = {FACE_LANDMARKS}
FACE_DIMS = {FACE_DIMS}
MAX_HANDS = {MAX_HANDS}
MAX_FACES = {MAX_FACES}


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.use_pose = {use_pose}
        self.use_hands = {use_hands}
        self.use_face = {use_face}
        
        # Инициализация MediaPipe с максимальными параметрами качества
        if self.use_pose:
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.mp_pose = None
        
        if self.use_hands:
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=MAX_HANDS,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.mp_hands = None
        
        if self.use_face:
            self.mp_face = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                refine_landmarks=True,
                max_num_faces=MAX_FACES,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.mp_face = None

    def execute(self, requests):
        responses = []
        for request in requests:
            # Получаем входной тензор: (B, H, W, 3) uint8 RGB
            x_u8 = pb_utils.get_input_tensor_by_name(request, "INPUT__0").as_numpy()
            B = x_u8.shape[0]
            
            # Инициализируем выходные массивы с NaN
            pose_out = np.full((B, POSE_LANDMARKS, POSE_DIMS), np.nan, dtype=np.float32)
            hands_out = np.full((B, MAX_HANDS, HAND_LANDMARKS, HAND_DIMS), np.nan, dtype=np.float32)
            face_out = np.full((B, MAX_FACES, FACE_LANDMARKS, FACE_DIMS), np.nan, dtype=np.float32)
            
            # Обрабатываем каждый кадр в батче
            for i in range(B):
                img = x_u8[i]  # H, W, C uint8 RGB
                
                if self.use_pose and self.mp_pose:
                    res_pose = self.mp_pose.process(img)
                    if res_pose.pose_landmarks:
                        for j, lm in enumerate(res_pose.pose_landmarks.landmark):
                            pose_out[i, j] = [lm.x, lm.y, lm.z, lm.visibility]
                
                if self.use_hands and self.mp_hands:
                    res_hands = self.mp_hands.process(img)
                    if res_hands.multi_hand_landmarks:
                        for h, hand in enumerate(res_hands.multi_hand_landmarks):
                            if h >= MAX_HANDS:
                                break
                            for j, lm in enumerate(hand.landmark):
                                hands_out[i, h, j] = [lm.x, lm.y, lm.z]
                
                if self.use_face and self.mp_face:
                    res_face = self.mp_face.process(img)
                    if res_face.multi_face_landmarks:
                        for f, face in enumerate(res_face.multi_face_landmarks):
                            if f >= MAX_FACES:
                                break
                            for j, lm in enumerate(face.landmark):
                                face_out[i, f, j] = [lm.x, lm.y, lm.z]
            
            # Создаем выходные тензоры (всегда присутствуют, даже если модель отключена)
            # Это требуется для совместимости с config.pbtxt, где все выходы объявлены
            outputs = [
                pb_utils.Tensor("POSE", pose_out),
                pb_utils.Tensor("HANDS", hands_out),
                pb_utils.Tensor("FACE", face_out),
            ]
            
            responses.append(pb_utils.InferenceResponse(output_tensors=outputs))
        
        return responses
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(model_code)
    
    print(f"✓ Python backend model created: {output_path}")


def create_triton_model(
    output_dir: str,
    use_pose: bool = True,
    use_hands: bool = True,
    use_face: bool = True,
):
    """
    Создает Python backend модель для Triton.
    MediaPipe не может быть экспортирован в ONNX из-за использования Python кода,
    поэтому используем Python backend напрямую.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "model.py")
    
    print(f"Creating Triton Python backend model: {model_path}")
    print(f"  Use pose: {use_pose}")
    print(f"  Use hands: {use_hands}")
    print(f"  Use face: {use_face}")
    
    create_triton_python_model(
        output_path=model_path,
        use_pose=use_pose,
        use_hands=use_hands,
        use_face=use_face,
    )
    
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Create Triton Python backend model for MediaPipe")
    parser.add_argument("--output-dir", required=True, help="Output directory for Python backend model (e.g., .../mediapipe_landmarks/1/)")
    parser.add_argument("--use-pose", action="store_true", default=True, help="Include pose model")
    parser.add_argument("--use-hands", action="store_true", default=True, help="Include hands model")
    parser.add_argument("--use-face", action="store_true", default=True, help="Include face model")

    args = parser.parse_args()

    create_triton_model(
        output_dir=args.output_dir,
        use_pose=args.use_pose,
        use_hands=args.use_hands,
        use_face=args.use_face,
    )


if __name__ == "__main__":
    main()

