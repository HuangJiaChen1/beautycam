import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For static images:
IMAGE_FILES = ['imgs/face_celian.jpg']
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image = cv2.resize(image, (8000, 8000), interpolation=cv2.INTER_LANCZOS4)
        # Convert the BGR image to RGB before processing.
        image = cv2.flip(image, 1)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            print('face_landmarks:', face_landmarks)

            # Draw the mesh tessellation
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())

            # Draw the contours
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())

            # Draw the irises
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())

            # Annotate each point with its index number
            for i, landmark in enumerate(face_landmarks.landmark):
                # Convert normalized coordinates to pixel values
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                # Draw the point number next to each point
                cv2.putText(annotated_image, str(i), (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                # cv2.circle(annotated_image,(x, y), 1, (0, 255, 0), 2)
        # Show the annotated image with point numbers
        cv2.imshow('image', annotated_image)
        cv2.waitKey(0)

        # Optionally save the annotated image
        cv2.imwrite('points_flipped.jpg', annotated_image)
