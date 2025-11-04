import cv2
import mediapipe as mp


mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

    connections = mp_face_mesh.FACEMESH_NOSE
    print(connections)
    full_list = []
    for (a,b) in connections:
        if a not in full_list:
            full_list.append(a)
        if b not in full_list:
            full_list.append(b)

    print(full_list)