import numpy as np
import cv2
import paddlehub as hub
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 创建VideoCapture，传入0即打开系统默认摄像头
vc = cv2.VideoCapture(0)

# 创建module
module = hub.Module(name='ace2p')

n = 0
count = 0
sum_count = 40
colors = np.array([
    # [139, 0, 0],
    # [255, 255, 0],  # yellow
    [255, 215, 0],  # gold
    # [255, 105, 180],  # hotpink
    # [255, 0, 255],  # fuchsia
    [224, 255, 255],  # lightcyan
    [139, 69, 19],  # saddlebrown
    [128, 128, 0],  # olive
    [124, 252, 0],  # lawngreen
    # [0, 0, 255],  # blue
    [0, 128, 0],  # green
])
len_colors = colors.shape[0]

while True:
    # 读取一帧，read()方法是其他两个类方法的结合，具体文档
    # ret为bool类型，指示是否成功读取这一帧
    ret, frame = vc.read()
    frame = cv2.flip(frame, 1)
    # 抠出人脸
    img = module.segmentation(images=[frame], use_gpu=True)[0]['data']
    img[img != 2] = 1
    img[img == 2] = 0

    img = img[:, :, np.newaxis]
    img = np.repeat(img, repeats=3, axis=2)

    frame = np.multiply(frame, img)

    img = np.abs(img - 1)
    img = np.multiply(img, colors[n])

    frame += img.astype('uint8')
    # 若没有按下q键，则每1毫秒显示一帧
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('frame', frame)
    count += 1
    if count == sum_count:
        count = 0
        n += 1
        if n == 5:
            sum_count = 60
        else:
            sum_count = 40
        if n == len_colors:
            n = 0
# 所有操作结束后不要忘记释放
vc.release()
cv2.destroyAllWindows()
