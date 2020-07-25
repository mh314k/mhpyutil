import numpy as np
import cv2
from skimage.metrics import structural_similarity


def stackImage(images: list, gray_scale: bool = False, win_width: int = 860, win_height: int = None,
               show_process: bool = False, bg_color: tuple = (255, 255, 255)) -> np.ndarray:
    """
    a function that make a stack Image.
    :param images: list of images as a nested list.
    :param gray_scale: gray scale property of output image.
    :param win_width: width of output image.
    :param win_height: height of output image
    :param show_process: set True to show the process in terminal. not recomended for videos.
    :param bg_color: image background color
    :return:
    """

    # row and columns count and every row's ratio sum calculating.
    # ___________________________________________________________.
    # rows count.
    rows = len(images)
    # columns count.
    cols = []
    # weighted width of every image(width/height).
    w = []
    if show_process: print("Calculating sizes...")
    for image_row in images:
        cols.append(len(image_row))
        _w = 0
        for image in image_row:
            _w += image.shape[1] / image.shape[0]
        w.append(_w)
    # calculating height of every image.
    height = int(rows * (win_width / max(w))) if (win_height is None) else win_height
    # height of every image.
    rows_height = int(height / rows)
    if show_process: print("Done!")
    if show_process: print("Resizing images...")
    resized_images = []
    for image_row in images:
        resized_imge_row = []
        for image in image_row:
            if len(image.shape) == 3:
                if gray_scale:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                if not gray_scale:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = cv2.resize(image, (int(image.shape[1] * rows_height / image.shape[0]), rows_height))
            resized_imge_row.append(image)
        resized_images.append(resized_imge_row)
    if show_process: print("Done!")
    if show_process: print("Attaching Images...")
    filed_width = []
    for image_row in resized_images:
        fw = 0
        for image in image_row:
            fw += image.shape[1]
        filed_width.append(fw)
    reminded_width = [win_width - fw for fw in filed_width]
    for i in range(rows):

        if reminded_width[i] > 0:
            b = np.ones((rows_height, reminded_width[i], 3), dtype=np.uint8) * 255
        if gray_scale and reminded_width[i] > 0:
            b = np.ones((rows_height, reminded_width[i], 3), dtype=np.uint8) * 255
            b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        if reminded_width[i] > 0:
            b[:] = bg_color
            resized_images[i].append(b)
    first_row = resized_images[0]
    first_image = first_row[0]
    for image in first_row[1:]:
        first_image = np.concatenate((first_image, image), axis=1)
    result = first_image
    for image_row in resized_images[1:]:
        first_in_row = image_row[0]
        for image in image_row[1:]:
            first_in_row = np.concatenate((first_in_row, image), axis=1)
        result = np.concatenate((result, first_in_row), axis=0)
    if show_process: print("Done!")
    if show_process: print("Finished!")
    return result


cap = cv2.VideoCapture(0)
while 1:
    # noinspection PyBroadException
    try:
        _, frame = cap.read()
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cframe = cv2.Canny(frame, 100, 200)
        fframe = cv2.Canny(gframe, 100, 200)
        hframe = cv2.bitwise_and(fframe, cframe)

        cv2.putText(hframe, f"SSIM: {structural_similarity(gframe, fframe):1.5}", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 0, 0))
        cv2.imshow("asd", stackImage(
            [[frame, cframe],
             [gframe, fframe],
             [hframe]],
            win_width=860, bg_color=(100, 150, 200)))
    except:
        print(f"Error")
        cap.release()
        cv2.destroyAllWindows()
        break
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
