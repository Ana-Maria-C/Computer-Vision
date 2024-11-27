import cv2


def resize_with_aspect_ratio(img, scale):

    (h, w) = img.shape[:2]
    new_width = int(w * scale)
    new_height = int(h * scale)

    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return resized


def resize_alter_aspect_ratio(img, new_width, new_height):

    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA if new_width < img.shape[1] or new_height < img.shape[0] else cv2.INTER_LINEAR)
    return resized


