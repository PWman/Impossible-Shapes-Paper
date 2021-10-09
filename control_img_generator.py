import os
import config
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from more_utils import set_seed
from skimage.draw import circle_perimeter, polygon_perimeter
from skimage.morphology import dilation, disk

set_seed(0)


def generate_circle():
    radius = np.random.randint(20, 100)
    x_centre = np.random.randint(radius + 5, 224 - radius - 5)
    y_centre = np.random.randint(radius + 5, 224 - radius - 5)
    rr, cc = circle_perimeter(x_centre, y_centre, radius)
    return rr, cc


def generate_square():
    square_len = np.random.randint(35, 177)
    x_corner = np.random.randint(5, 224 - square_len - 5)
    y_corner = np.random.randint(5, 224 - square_len - 5)
    poly_corners = np.array((
        (y_corner + square_len, x_corner),
        (y_corner, x_corner),
        (y_corner, x_corner + square_len),
        (y_corner + square_len, x_corner + square_len)
    ))
    rr, cc = polygon_perimeter(poly_corners[:, 0],
                               poly_corners[:, 1])
    return rr, cc


def generate_random_shape_arr(func):
    img = np.zeros((224, 224, 3))
    for i in range(4):
        rr, cc = func()
        img[rr, cc, :] = (1, 1, 1)
    dilated_img = dilation(img)
    return 1-dilated_img


def save_control_shapes():
    def save_shapes(func,shape_name):
        save_path = os.path.join(save_basedir, shape_name)
        config.check_make_dir(save_path)
        for i in range(40):
            arr = generate_random_shape_arr(func)
            img = Image.fromarray(np.uint8(arr * 255))
            img.save(os.path.join(save_path, f"{shape_name}{i}.bmp"))

    save_basedir = os.path.join(config.original_dir, "Control")
    config.check_make_dir(save_basedir)
    save_shapes(generate_circle, "Circle")
    save_shapes(generate_square, "Square")


if __name__ == "__main__":
    save_control_shapes()