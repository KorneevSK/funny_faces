# Googly Eyes App

An entertaining web application that adds googly eyes to faces in your images. The app consists of a FastAPI backend for image processing and a Streamlit frontend for user interaction. The entire application is containerized using Docker for easy deployment.

### Features
* **Face and Eye Detection**: Uses OpenCV's Haar Cascade classifiers to detect faces and eyes.
* **Googly Eyes Overlay**: Adds googly eyes to detected eyes with randomized pupil positions for a fun effect.
* **Parallel Processing**: Utilizes joblib for parallel processing to enhance performance.
* **User-Friendly Interface**: Streamlit frontend for easy image upload and display.
* **Containerized Deployment**: Dockerized application for seamless deployment.

### Architecture
* **Frontend**: Streamlit app (app.py) provides the user interface.
* **Backend**: FastAPI app (service/main.py) processes images.
* **Image Processing**: service/image_processing.py contains the image processing logic.
* **Docker**: Docker Compose orchestrates the frontend and backend services.

The app is available at:
http://funnyfaces.freemyip.com/
