import base64
from functools import wraps
import json
import socket
from time import sleep

import cv2


class Client:
    def __init__(self, address, port):
        self.address = address
        self.port = port
        self.conn = socket.socket()
        self.is_reconnecting = False

    def connect(self):
        count = 0
        while True:
            try:
                self.conn.connect((self.address, self.port))
                self.is_reconnecting = False
                print("Connection success after %d reconnection(s) at %s:%d" % (count, self.address, self.port))
                return
            except Exception as e:
                self.is_reconnecting = True
                self.conn = socket.socket()
                count += 1
                print(str(e) + " at %s:%d ...... %d" % (self.address, self.port, count))
                sleep(1)

    # def send_to_video_server(self, camera_id, data_type, data_byte):
    #     data_byte = "header".encode() + (len(data_byte) + 8).to_bytes(length=4, byteorder="little", signed=True) \
    #                 + camera_id.to_bytes(length=4, byteorder="little", signed=True) \
    #                 + data_type.to_bytes(length=4, byteorder="little", signed=True) + data_byte
    #     try:
    #         self.conn.send(data_byte)
    #     except Exception as e:
    #         print(str(e) + ": Reconnecting " + str(self.address) + ":" + str(self.port))
    #         self.conn = socket.socket()
    #         self.connect()

    # def send_to_video_server2(self, msg_id, camera_key, data_byte):
    #     data_byte = (len(data_byte)+4).to_bytes(length=4, byteorder="big", signed=False) \
    #         + msg_id.to_bytes(length=4, byteorder="big", signed=False) \
    #         + camera_key.to_bytes(length=4, byteorder="big", signed=False) \
    #         + data_byte
    #     try:
    #         self.conn.send(data_byte)
    #     except Exception as e:
    #         print(str(e) + ": Reconnecting " + str(self.address) + ":" + str(self.port))
    #         self.conn = socket.socket()
    #         self.connect()

    def send(self, key, frame, text):
        _, frame_encode = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
        frame_base64 = str(base64.b64encode(frame_encode))[2:-1]
        # frame_decode = cv2.imdecode(frame_encode, cv2.IMREAD_COLOR)
        try:
            self.conn.send(self.encode_to_bytes(key, frame_base64, 120))
            self.conn.send(self.encode_to_bytes(key, text, 121))
        except Exception as e:
            print(str(e) + ": Reconnecting " + str(self.address) + ":" + str(self.port))
            self.conn = socket.socket()
            self.connect()

    @staticmethod
    def encode_to_bytes(key, data, msg_id):
        dict_data = {
            'key': key,
            'data': data,
        }
        json_bytes = json.dumps(dict_data).encode()
        data_bytes = len(json_bytes).to_bytes(length=4, byteorder="big", signed=False) \
            + msg_id.to_bytes(length=4, byteorder="big", signed=False) \
            + json_bytes
        return data_bytes

    # def send_to_data_center(self, data):
    #     data_json = json.dumps(data)
    #     data_byte = data_json.encode()
    #     data_byte = "header".encode() + len(data_byte).to_bytes(length=4, byteorder="little", signed=True) + data_byte
    #     try:
    #         self.conn.send(data_byte)
    #         print("send register information successfully")
    #     except Exception as e:
    #         print(str(e) + ": Reconnecting " + str(self.address) + ":" + str(self.port))
    #         self.conn = socket.socket()
    #         self.connect()


if __name__ == "__main__":
    client = Client("localhost", 9090)
    client.send("hello")
