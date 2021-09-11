import mss
import time
import cv2
import numpy as np
from ppadb.client import Client
from time import time, sleep
from io import BytesIO
from PIL import Image
import argparse


parser = argparse.ArgumentParser(description='Auto Fishing ROX. On 720p only.')
parser.add_argument('--device', type=str, default='emulator-5556', help='ADB Connect to device')
parser.add_argument('--bait_number', type=int, default=50, help='Number of Bait used')
parser.add_argument('--monitor_number', type=int, default=1, help='Monitor that the program lived')
parser.add_argument('--offset_x', type=int, default=0, help='Offset X of the game (default is top left)')
parser.add_argument('--offset_y', type=int, default=33, help='Offset Y of the game (default is top left)')

args = parser.parse_args()
print(args.device)
print(args.bait_number)

client = Client(host="127.0.0.1", port=5037)
device = client.device(f"{args.device}")
    
print(device)

cast_img = cv2.imread('cast.png', cv2.IMREAD_UNCHANGED)
cast_img_gray = cv2.cvtColor(cast_img, cv2.COLOR_BGR2GRAY)
w_cast = cast_img.shape[1]
h_cast = cast_img.shape[0]

reel_img = cv2.imread('reel.png', cv2.IMREAD_UNCHANGED)
w_reel = reel_img.shape[1]
h_reel = reel_img.shape[0]

bait_number = args.bait_number
clicked_cast = 0
clicked_reel = 0

with mss.mss() as sct:
    # Part of the screen to capture
    monitor_number = args.monitor_number
    mon = sct.monitors[monitor_number]
    monitor = {
        "left": mon["left"] + args.offset_x + 950,  # 100px from the left
        "top": mon["top"] + args.offset_y + 420,  # 100px from the top
        "width": 200,
        "height": 200,
        "mon": monitor_number,
    }

    while "Screen capturing":
        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))
        screen_img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cast_result = cv2.matchTemplate(screen_img_gray, cast_img_gray, cv2.TM_CCOEFF_NORMED)
        yloc, xloc = np.where(cast_result >= .80)
        rectangles = []
        for (x, y) in zip(xloc, yloc):
            rectangles.append([int(x), int(y), int(w_cast), int(h_cast)])
            rectangles.append([int(x), int(y), int(w_cast), int(h_cast)])
        rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.8)
        cv2.imshow("OpenCV/Numpy normal", img)
        for (x, y) in zip(xloc, yloc):
            cv2.rectangle(img, (x, y), (x + w_cast, y + h_cast), (0, 255, 255), 2)
            if(clicked_cast == 0):
                device.shell(f'input tap 1090 530')
                sleep(0.1)
                clicked_cast = 1
                clicked_reel = 0
            cv2.imshow("OpenCV/Numpy normal", img)
            
        reel_result = cv2.matchTemplate(img, reel_img, cv2.TM_CCOEFF_NORMED)
        yloc, xloc = np.where(reel_result >= .70)
        rectangles = []
        for (x, y) in zip(xloc, yloc):
            rectangles.append([int(x), int(y), int(w_reel), int(h_reel)])
            rectangles.append([int(x), int(y), int(w_reel), int(h_reel)])
        rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.8)
        cv2.imshow("OpenCV/Numpy normal", img)
        for (x, y) in zip(xloc, yloc):
            cv2.rectangle(img, (x, y), (x + w_reel, y + h_reel), (0, 255, 255), 2)
            if(clicked_reel == 0):
                device.shell(f'input tap 1090 530')
                sleep(0.1)
                clicked_cast = 0
                clicked_reel = 1
                bait_number -= 1
            cv2.imshow("OpenCV/Numpy normal", img)

        cv2.imshow("OpenCV/Numpy normal", img)

        # Press "q" to quit
        if (cv2.waitKey(25) & 0xFF == ord("q")) or bait_number == 0:
            cv2.destroyAllWindows()
            break