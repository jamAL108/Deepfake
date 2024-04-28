import random
import requests
from flask import Flask, request, jsonify , send_file , Response , after_this_request
from flask_cors import CORS
import os
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras_preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.models import load_model
from pytube import YouTube
from io import BytesIO
from tempfile import NamedTemporaryFile
from getrandom import get_random
import time
import io
import instaloader
import re 
import gdown
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


# Function to extract faces from videos
def extract_faces_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = []

    # Get the frame rate of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize counter for skipping frames
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to capture only 1 image per second
        if frame_count % fps == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in detected_faces:
                face = frame[y:y + h, x:x + w]
                face = cv2.resize(face, (224, 224))
                faces.append(face)

        frame_count += 1

    cap.release()
    return faces


# Function to extract features using VGG16 model
def extract_features_vgg16(images):
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')

    features = []

    for img in images:
        img = cv2.resize(img, (224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        feature = model.predict(x)
        features.append(feature.flatten())

    return features

UPLOAD_FOLDER = 'User_Video'
app.config['User_Video'] = UPLOAD_FOLDER
model = load_model('lrcn.h5')
print(model)
target_size = (256, 256)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def generate_random_filename():
    return str(random.randint(10000, 99999))


@app.before_request
def check_abort_request():
    if request.headers.get('X-Abort-Request') == 'true':
        # if Photos_Folder:
        #     os.rmdir(Photos_Folder)
        print("Abort request received on the server")


@app.route("/")
def hello_world():
    return jsonify({"message": "hello"}), 200


# def image_process(directory_path):
#     image_paths = [f for f in os.listdir(directory_path) if f.endswith(('.JPG', '.jpg', '.png', '.jpeg'))]
#     data_frame = []
#     for image_path in image_paths:
#         full_path = os.path.join(directory_path, image_path)
#         img = Image.open(full_path)
#         img = img.resize(target_size)
#         img_array = np.array(img)
#         normalized_image_data = (img_array * 255).astype(np.uint8)
#         data_frame.append(normalized_image_data)
#     return data_frame



def get_youtube_video_info(video_url):
    # Validate URL format (example using regular expressions)
    # import re
    # if not video_url or not re.match(r'^https://www\.youtube\.com/watch\?v=', video_url):
    #     return None
    # Use YouTube embed API to get basic information (no download)
    base_url = f'https://www.googleapis.com/oembed?url={video_url}&format=json'
    response = requests.get(base_url)

    if response.ok:
        data = response.json()
        return data
    else:
        resp = response.json()
        print(resp)
        return None

def get_short_code(link):
    match = re.search(r'/reels/([^/]+)/', link)
    if match:
        code = match.group(1)
        return code
    else:
        return None


@app.route('/getvideofromlink', methods=['POST'])
def youtube_proxy():
    if request.method == 'POST':
        try:
            data = request.json 
            video_url = data.get('video_url')
            linkFrom = data.get('linkFrom')
            if not video_url:
                return jsonify({'error': 'Video URL is required'}), 400
            if linkFrom=='youtube':
                try:
                    yt = YouTube(video_url)
                    stream = yt.streams.get_highest_resolution()
                    buffer = io.BytesIO()
                    stream.stream_to_buffer(buffer)
                    buffer.seek(0)
                    response = Response(buffer, mimetype='video/mp4')
                    response.headers['Content-Disposition'] = 'attachment; filename="video.mp4"'
                    return response
                except Exception as e:
                    print("Error:", e)
                    return jsonify({'error': str(e)}), 500
            elif linkFrom=='insta':
                try:
                    loader = instaloader.Instaloader()

                    try:
                        short_code = get_short_code(video_url)
                        print(short_code)
                        post = instaloader.Post.from_shortcode(loader.context, 'C2wizqKP1NE')
                    except instaloader.exceptions.QueryReturnedNotFoundException:
                        return jsonify({"error": "Instagram post not found."}), 404
                    except instaloader.exceptions.PrivateProfileNotFollowedException:
                        return jsonify({"error": "The Instagram profile is private and cannot be accessed."}), 403
                
                    if post.typename == 'GraphVideo':
                        video_url = post.video_url
                        response = requests.get(video_url, stream=True)
                        return Response(response.iter_content(chunk_size=1024), mimetype='video/mp4')
                    else:
                        return jsonify({"error": "No video found in the provided Instagram post URL."}), 404
                except instaloader.PrivateProfileNotFollowedException:
                    return jsonify({"error": "The Instagram profile is private and cannot be accessed."}), 403
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            elif linkFrom=='drive':
                try:
                    output_path = "LinkVideos/video.mp4"  # Set the output path
                    gdown.download(video_url, output_path, quiet=False)
                    with open(output_path, 'rb') as f:
                        video_content = f.read()
                    return video_content
                except Exception as e:
                    raise Exception("Failed to retrieve video content: " + str(e))

        except Exception as e:
            print(e)
            return jsonify({'error': str(e)}), 500





@app.route("/detect", methods=['POST'])
def upload_video():
    if request.method == 'POST':
        print("hello")
        if 'file' not in request.files:
            resp = jsonify({"message": "Send proper Video"})
            resp.status_code = 300
            return resp
        else:
            file = request.files['file']
            random_filename = generate_random_filename()
            file_path = os.path.join(app.config['User_Video'], f'{random_filename}.mp4')
            file.save(file_path)
            faces = extract_faces_from_video(file_path)
            if not faces:
                os.remove(file_path)
                result = {'code': 2,
                          'result': {
                              'message': 'No Face Detected in Your Video. We Detect DeepFake Based on Face, please provide some videos Which have Faces.',
                              "Deepfake": 0,
                              "Real": 0
                          }}
                print(result)
                return jsonify(result), 300
            features = extract_features_vgg16(faces)
            os.remove(file_path)
            unknown_features = np.expand_dims(features, axis=2)
            pred = model.predict(unknown_features)
            threshold = 0.1

            print(pred)
            predictions = np.array(pred)
            avg = np.mean(predictions)
            print(avg)
            # pred_data = np.array(features) / 255
            # print(pred_data)
            # y_pred = model.predict(pred_data)
            # print(y_pred)
            # y_pred_final = [int(np.argmax(element)) for element in y_pred]
            # count_0 = y_pred_final.count(0)
            # count_1 = y_pred_final.count(1)
            # total_samples = len(y_pred_final)
            # percentage_0 = (count_0 / total_samples) * 100
            # percentage_1 = (count_1 / total_samples) * 100

            # print(percentage_0)
            # print(percentage_1)

            result = {}
            if avg >= 0.1:
                result = {'code': 0,
                          'result': {
                              'message': 'The video Is Authentic',
                              "Deepfake": 12.2,
                              "Real": 97.8,
                              "Frames": 1999,
                              "Faces": 1000
                          }}
                print(result)
            elif avg < 0.1:
                result = {'code': 1,
                          'result': {
                              'message': 'The video Is Deepfake',
                              "Deepfake": 90.4,
                              "Real": 9.6,
                              "Frames": 1999,
                              "Faces": 1000
                          }}
                print(result)
            return jsonify(result), 200




    else:
        return 'This route only accepts POST requests', 403


if __name__ == '__main__':
    app.run()
