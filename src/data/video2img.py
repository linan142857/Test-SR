import os
import cv2
frm = 1005000
nums = 1000
count = 1
source = '/home/yulin/Downloads/Trespass.Against.Us.2016.BluRay.720p.DTS.x264-CHD/Trespass.Against.Us.2016.BluRay.720p.DTS.x264-CHD.mkv'
target = '/home/yulin/Documents/SR/Test-SR/data/video/val'
cap = cv2.VideoCapture(source)
if cap.isOpened():
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))

for i in range(frm):
    ret, frame = cap.read()
while ret and count <= nums:
    ret, frame = cap.read()
    cv2.imwrite(os.path.join(target, '%06d.png' % count), frame)
    count += 1
    print('Index at No.' + str(count) + ' Frames.')
cap.release()





















