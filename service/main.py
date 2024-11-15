# googly_eyes_service/main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO
from .image_processing import add_googly_eyes

app = FastAPI()

@app.get("/")
async def read_root() -> dict:
    """
    Reads the root endpoint.

    Returns
    -------
    dict
        A dictionary containing a welcome message.
    """
    return {"message": "Welcome to the Googly Eyes API!"}

@app.post("/apply_googly_eyes/")
async def apply_googly_eyes(file: UploadFile = File(...)) -> StreamingResponse:
    """
    Applies googly eyes to an uploaded image file.

    Parameters
    ----------
    file : UploadFile
        The uploaded image file to which googly eyes will be applied.
    Returns
    -------
    StreamingResponse
        A streaming response containing the processed image with googly eyes in PNG format.
    """
    # Read the uploaded image file
    image = Image.open(file.file)

    # Apply googly eyes
    googly_image = add_googly_eyes(image)

    # Save the processed image to a byte array
    img_byte_arr = BytesIO()
    googly_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")
