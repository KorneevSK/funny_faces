from typing import Tuple
from joblib import Parallel, delayed
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Load Haar cascade models for face and eye detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

def process_eye(face_image: np.ndarray, eye_coords: Tuple) -> np.ndarray:
    """
    Processes an eye region in a face image by overlaying a googly eye.

    Parameters
    ----------------
    face_image : numpy.ndarray
        The image of the face in which the eye is to be processed.
    eye_coords : tuple
        A tuple containing the coordinates and size of the eye region (ex, ey, ew, eh).
    
    Returns
    ----------------
    numpy.ndarray
        The face image with the googly eye overlay applied to the specified eye region.
    """
    ex, ey, ew, eh = eye_coords

    # Create a googly eye with random size and pupil position
    googly_eye = Image.new("RGBA", (ew, eh), (0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(googly_eye)
    draw.ellipse([0, 0, ew, eh], fill=(255, 255, 255, 255))  # White part of the eye

    # Randomize pupil size and position within the eye area
    pupil_radius = ew // 3
    pupil_offset_x = np.random.randint(-pupil_radius // 2, pupil_radius // 2)
    pupil_offset_y = np.random.randint(-pupil_radius // 2, pupil_radius // 2)
    draw.ellipse(
        [
            pupil_offset_x + ew // 3,
            pupil_offset_y + eh // 3,
            pupil_offset_x + ew // 3 + pupil_radius,
            pupil_offset_y + eh // 3 + pupil_radius,
        ],
        fill=(0, 0, 0, 255),  # Black pupil
    )

    # Convert googly eye to OpenCV format with alpha channel
    googly_eye_cv = np.array(googly_eye)
    googly_eye_bgra = cv2.cvtColor(googly_eye_cv, cv2.COLOR_RGBA2BGRA)

    # Define the region in the face image where the googly eye will be placed
    eye_region = face_image[ey : ey + eh, ex : ex + ew]

    # Ensure the sizes match for overlay; resize if necessary
    if eye_region.shape[0:2] != googly_eye_bgra.shape[0:2]:
        googly_eye_bgra = cv2.resize(
            googly_eye_bgra, (eye_region.shape[1], eye_region.shape[0])
        )

    # Use alpha blending to overlay the googly eye with transparency
    alpha_mask = googly_eye_bgra[:, :, 3] / 255.0  # Alpha channel as mask
    for c in range(3):  # Iterate over B, G, R channels
        eye_region[:, :, c] = (
            1 - alpha_mask
        ) * eye_region[:, :, c] + alpha_mask * googly_eye_bgra[:, :, c]

    # Update the eye region in the face image
    face_image[ey : ey + eh, ex : ex + ew] = eye_region

    return face_image

def process_face(open_cv_image: np.ndarray, face_coords: Tuple) -> Tuple:
    """
    Processes a face in an image by extracting the face region, detecting eyes, 
    and processing the eyes in parallel.

    Parameters
    ----------------
    open_cv_image : numpy.ndarray
        The input image in which the face is to be processed.
    face_coords : tuple
        A tuple (x, y, w, h) representing the coordinates and size of the face 
        in the image.

    Returns
    ----------------
    tuple
        A tuple (x, y, w, h, face_image) where (x, y, w, h) are the coordinates 
        and size of the face, and face_image is the processed face region.
    """
    x, y, w, h = face_coords

    # Extract face region
    face_image = open_cv_image[y : y + h, x : x + w].copy()

    # Detect eyes within the face region
    eyes = eye_cascade.detectMultiScale(face_image)

    # Sort detected eyes by size and take the two largest
    if len(eyes) > 2:
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

    # Process eyes in parallel
    if len(eyes) > 0:
        face_image = Parallel(n_jobs=-1)(
            delayed(process_eye)(face_image, eye_coords) for eye_coords in eyes
        )[0]

    return (x, y, w, h, face_image)

def add_googly_eyes(image):
    """
    Detects faces and eyes in an image and overlays googly eyes on each detected eye.
    Randomizes the size and position of pupils for added humor.

    Parameters
    ----------------
    image : PIL.Image
        The input image in which googly eyes will be added.

    Returns
    ----------------
    PIL.Image
        The image with googly eyes applied.
    """
    # Convert the PIL image to an OpenCV-compatible format
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        open_cv_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Process faces in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_face)(open_cv_image, face_coords) for face_coords in faces
    )

    # Overlay processed faces onto the original image
    for (x, y, w, h, face_image) in results:
        open_cv_image[y : y + h, x : x + w] = face_image

    # Convert back to Pillow format for easy display and return
    result_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_image)
