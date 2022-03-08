import cv2
import gc
import multiprocessing
from multiprocessing import Process, Manager
import subprocess
from time import sleep


def readVideo(read_stack) -> None:
    cap = cv2.VideoCapture('rtsp://admin:c3i123456@192.168.22.64:554/Streaming/Channels/101')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sizeStr = str(size[0]) + 'x' + str(size[1])
    print(sizeStr)
    while True:
        ret, frame = cap.read()
        if frame is not None:
            if len(read_stack) == 0:
                read_stack.append(frame)
                # del read_stack[:]
                # gc.collect()


def processFrame(read_stack, write_queue) -> None:
    while True:
        if len(read_stack) != 0:
            try:
                frame = read_stack.pop()
            except Exception as e:
                print(e.args)
                sleep(0.1)
                frame = read_stack.pop()
            cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
            cv2.imshow("camera", frame)
            key = cv2.waitKey(1) & 0xff
            if key == ord("q"):
                break
            write_queue.put(frame)
    cv2.destroyAllWindows()


def writeFrame(write_queue):
    rtmp_url = "rtmp://localhost:1935/live/test"
    # rtmp_url = "rtmp://localhost:1935/live/rfBd56ti2SMtYvSgD5xAV0YU99zampta7Z7S575KLkIZ9PYk"
    command = ['/usr/bin/ffmpeg',
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24',
               # '-s', '{}x{}'.format(2560, 1920),
               '-s', '{}x{}'.format(1280, 720),
               # '-s', '{}x{}'.format(1920, 1080),
               '-r', '50',
               '-i', '-',
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               '-preset', 'ultrafast',
               '-f', 'flv',
               rtmp_url]

    pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)
    while True:
        frame = write_queue.get()
        pipe.stdin.write(frame.tobytes())
    


def main():
    read_stack = Manager().list()
    write_queue = Manager().Queue()
    read = Process(target=readVideo, args=(read_stack,))
    process = Process(target=processFrame, args=(read_stack, write_queue,))
    write = Process(target=writeFrame, args=(write_queue,))
    read.start()
    process.start()
    write.start()

    process.join()
    read.terminate()
    write.terminate()


if __name__ == "__main__":
    main()
