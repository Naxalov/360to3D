import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def deg2rad(d):
    return float(d) * np.pi / 180


def rotate_image(old_image):
    (old_height, old_width, _) = old_image.shape
    M = cv2.getRotationMatrix2D(((old_width - 1) / 2., (old_height - 1) / 2.), 270, 1)
    rotated = cv2.warpAffine(old_image, M, (old_width, old_height))
    return rotated


def xrotation(th):
    c = np.cos(th)
    s = np.sin(th)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def yrotation(th):
    c = np.cos(th)
    s = np.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def plot_equi(ex, ey, size):
    arr_ = np.zeros(size.shape)
    # arr_ = size.copy()
    arr_[ey, ex] = [255, 255, 255]
    plt.imshow(arr_)
    plt.show()


def ceiling(x, y):
    x_c = 10 / np.tan(x) * np.cos(y)

    plt.plot(x, y)
    plt.imshow()


def render_image_np(theta0, phi0, fov_h, fov_v, width, img):
    """
    theta0 is pitch
    phi0 is yaw
    render view at (pitch, yaw) with fov_h by fov_v
    width is the number of horizontal pixels in the view
    """
    m = np.dot(yrotation(phi0), xrotation(theta0))

    (base_height, base_width, _) = img.shape

    height = int(width * np.tan(fov_v / 2) / np.tan(fov_h / 2))

    new_img = np.zeros((height, width, 3), np.uint8)

    DI = np.ones((height * width, 3), np.int)
    trans = np.array([[2. * np.tan(fov_h / 2) / float(width), 0., -np.tan(fov_h / 2)],
                      [0., -2. * np.tan(fov_v / 2) / float(height), np.tan(fov_v / 2)]])

    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    DI[:, 0] = xx.reshape(height * width)
    DI[:, 1] = yy.reshape(height * width)

    v = np.ones((height * width, 3), np.float)

    v[:, :2] = np.dot(DI, trans.T)
    v = np.dot(v, m.T)

    diag = np.sqrt(v[:, 2] ** 2 + v[:, 0] ** 2)
    theta = np.pi / 2 - np.arctan2(v[:, 1], diag)
    phi = np.arctan2(v[:, 0], v[:, 2]) + np.pi


    ey = np.rint(theta * base_height / np.pi).astype(np.int)
    ex = np.rint(phi * base_width / (2 * np.pi)).astype(np.int)
    ex[ex >= base_width] = base_width - 1
    ey[ey >= base_height] = base_height - 1
    ceiling(ex,ey)
    plot_equi(ex, ey, img)
    new_img[DI[:, 1], DI[:, 0]] = img[ey, ex]
    # x_p = np.resize(ex, (500, 500),np.float32)
    # y_p = np.resize(ey, (500, 500),np.float32)
    # presp = cv2.remap(img,x_p, y_p, cv2.INTER_CUBIC,  borderMode=cv2.BORDER_WRAP)
    return new_img


if __name__ == '__main__':
    path = os.getcwd()
    path = os.path.join(path, 'image/Seg_Color.jpg')
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 200))

    face_size = 300

    yaw = 0
    pitch = 40

    fov_h = 100
    fov_v = 100

    rimg = render_image_np(deg2rad(pitch), deg2rad(yaw),      deg2rad(fov_v), deg2rad(fov_h),          face_size, img)
    plt.imshow(rimg)
    print(rimg.shape)
    plt.show()

    # cv2.imwrite('rendered_image_%d_%d.bmp' % (pitch, yaw), rimg)
