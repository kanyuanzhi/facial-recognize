import base64
from multiprocessing import Manager

import json
from utils.socket_client import Client
import subprocess
import cv2

from utils.torch_utils import time_sync


# cap = cv2.VideoCapture('rtsp://admin:c3i123456@192.168.22.64:554/Streaming/Channels/101')
cap = cv2.VideoCapture(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(w, h, fps)

# rtmp_url = "rtmp://localhost:1935/live/test"
#     # rtmp_url = "rtmp://localhost:1935/live/rfBd56ti2SMtYvSgD5xAV0YU99zampta7Z7S575KLkIZ9PYk"
# command = ['/usr/bin/ffmpeg',
#             '-y',
#             '-f', 'rawvideo',
#             '-vcodec', 'rawvideo',
#             '-pix_fmt', 'bgr24',
#             # '-s', '{}x{}'.format(2560, 1920),
#             '-s', '{}x{}'.format(1280, 720),
#             # '-s', '{}x{}'.format(640, 480),
#             # '-s', '{}x{}'.format(1920, 1080),
#             '-r', '20',
#             '-i', '-',
#             '-c:v', 'libx264',
#             '-pix_fmt', 'yuv420p',
#             '-preset', 'ultrafast',
#             '-f', 'flv',
#             '-g', '5',
#             '-b',  '700000',
#             rtmp_url]

# pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

image_client = Client('localhost', 8888)
image_client.connect()

while True:
    ret, frame = cap.read()
    t1 = time_sync()
    # _, frame_encode = cv2.imencode('.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    _, frame_encode = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
    t2 = time_sync()
    frame_decode = cv2.imdecode(frame_encode, cv2.IMREAD_COLOR)
    t3 = time_sync()
    frame_base64 = str(base64.b64encode(frame_encode))[2:-1]

    # print(f't2-t1={t2-t1}   t3-t2={t3-t2}')
    # print(type(frame_encode))
    # print(type(frame))

    # print(len(frame.tobytes()))
    # print(len(frame_encode.tobytes()))

    # print(len(frame_base64))
    # print(type(frame_base64))
    data = {
        'key':"123",
        'data': frame_base64
    }
    data_bytes = json.dumps(data).encode()
    # image_client.send_to_video_server2(120, 123,frame.tobytes())
    image_client.send_to_video_server3(120, data_bytes)
    # pipe.stdin.write(frame.tobytes())
    # cv2.imshow("camera", frame_decode)
    # key = cv2.waitKey(100) & 0xff
    # if key == ord("q"):
    #     break
