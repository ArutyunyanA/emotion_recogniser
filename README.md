# Emotion Recognition with Facial Micro-Expressions

This project is designed to recognize emotions, including micro-expressions, using video files. The project is based on the works of Paul Ekman, including the concepts of micro-expressions and hidden emotions.

## Project Features
- Recognition of basic emotions: fear, sadness, joy, disgust, contempt, surprise, anger.
- Output of mixed emotions.
- Analysis of emotions in videos with original playback speed.
- Use of GPU (if available) to speed up calculations.

---

## Installation

### Cloning the repository
1. Clone this repository to your computer:

git clone https://github.com/your-username/emotion-recognition.git
cd emotion-recognition

### Creating a virtual environment
2. It is recommended to use a virtual environment to manage dependencies:

- For Linux/MacOS:

```
python3 -m venv venv
source venv/bin/activate

```
- For Windows:

```
python -m venv venv
venv\Scripts\activate
```

### Installing dependencies
3. Install all necessary packages from `requirements.txt`:

```
pip install -r requirements.txt
```

---

## Usage

### Prepare a video
Place your video file in the `videos` folder (or specify the path in the code).

### Running the project
To run emotion recognition, run the following command:

```
python face_recognition.py –input videos/your_video.mp4
```

### Result
The program will display the video with the recognized emotions and save the output as text (optional).

---

## Required software
- Python 3.7+

- TensorFlow
- OpenCV
- FER

3. Install dependencies

Install all required Python packages from the requirements.txt file:

pip install -r requirements.txt

If you have problems, check if you have installed dependencies:

```
pip freeze
```

---

## GPU acceleration
If you have a CUDA-enabled GPU, make sure TensorFlow is using it:

```
python -c “import tensorflow as tf; print(‘GPU available’ if tf.test.is_gpu_available() else ‘GPU not available’)”
```
To install TensorFlow with GPU support, run:

```
pip install tensorflow-gpu
```
---

## Issues and improvements
If you find a bug or have a suggestion for improving the project, create an [Issue](https://github.com/your-username/emotion-recognition/issues) or submit a Pull Request.

--
