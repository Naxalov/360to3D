import numpy as np
import matplotlib.pyplot as plt


def xyzpers(h_fov, v_fov, out_hw):
    out = np.ones((*out_hw, 3), np.float32)

    x_max = np.tan(h_fov / 2)
    y_max = np.tan(v_fov / 2)
    x_rng = np.linspace(-x_max, x_max, num=out_hw[1], dtype=np.float32)
    y_rng = np.linspace(-y_max, y_max, num=out_hw[0], dtype=np.float32)
    out[..., :2] = np.stack(np.meshgrid(x_rng, -y_rng), -1)

    return out


def cartesian_to_spherical(cartesian):
    X = cartesian[:, 0]
    Y = cartesian[:, 1]
    Z = cartesian[:, 2]

    rho = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    theta = np.arccos(Z / rho)
    phi = np.arctan2(Y, X)
    return np.vstack((rho, theta, phi)).T


def spherical_to_equirectangular(spherical):
    rho = spherical[:, 0]
    theta = spherical[:, 1]
    phi = spherical[:, 2]

    altitude = np.rad2deg(theta) - 90.0
    lattitude = np.rad2deg(phi)
    return np.vstack((lattitude, altitude)).T


def plot3d(x, y, z):
    # create 3d Axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, z)  # we've already pruned ourselves
    plt.show()


def uv2coor(uv, h, w):
    '''
    uv: ndarray in shape of [..., 2]
    h: int, height of the equirectangular image
    w: int, width of the equirectangular image
    '''
    u, v = np.split(uv, 2, axis=-1)
    coor_x = (u / (2 * np.pi) + 0.5) * w - 0.5
    coor_y = (-v / np.pi + 0.5) * h - 0.5

    return coor_x, coor_y


def xyz2uv(xyz):
    '''
    xyz: ndarray in shape of [..., 3]
    '''
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(x, z)
    c = np.sqrt(x ** 2 + z ** 2)
    v = np.arctan2(y, c)

    return np.concatenate([u, v], axis=-1)


def circle_line(a):
    samples = 200
    x = np.cos(np.linspace(-np.pi, np.pi, samples))
    y = np.sin(np.linspace(-np.pi, np.pi, samples))

    x_max = np.tan(7)
    y_max = np.tan(9)
    x_rng = np.linspace(-x_max, x_max, num=20, dtype=np.float32)
    y_rng = np.linspace(-y_max, y_max, num=20, dtype=np.float32)
    xx, yy = np.meshgrid(x_rng, -y_rng)
    fig = plt.figure(figsize=(10, 10))  # create a figure object
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    # ax.ylim(-1, 1)/

    ax.scatter(x, y)
    ax.scatter(xx, yy)
    plt.show()


def show_img(img):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    plt.show()


circle_line(1)
# samples = 500
# h = 50
# w = 100
# img = np.zeros((h, w, 3))
#
# xyz = xyzpers(10, 5, (50, 30))
# arr_ = np.ones((10, 10, 3))
# # arr_[]
# uv = xyz2uv(arr_)
#
# coor_xy = uv2coor(uv, h, w)
#
# coor_x, coor_y = uv2coor(uv, h, w)
#
# arr_x = coor_x[:, :, 0].astype(np.int)
# arr_y = coor_y[:, :, 0].astype(np.int)
# img[arr_y, arr_x] = [255, 0, 0]
# show_img(img)
