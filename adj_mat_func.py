import sys
import cv2
import numpy as np
import tensorflow as tf

"""
All function that are called with py_func to be use during training on adj-loss
"""


class SimpleAdjMat(object):
    """
    (0-1) Adjacency matrix for all batch images
    """

    def __init__(self, batch_size, pixel_distance=1):
        super().__init__()
        self.batch_size = batch_size
        self.pixel_distance = pixel_distance

    def adj_mat(self, y_true, y_pred):
        # Wraps np_adj_func method and uses it as a TensorFlow op.
        # Takes numpy arrays as its arguments and returns numpy arrays as
        # its outputs.
        return tf.py_func(self.np_adj_func, [y_true, y_pred], tf.float32)

    def np_adj_func(self, y_true, y_pred):
        """
        I nomi non sono significativi
        :param y_true: input image gray scale
        :param y_pred: not used
        :return:
        """

        # empty adj matrix
        adj_mat = np.zeros(shape=(108, 108))
        # one iteration for each batch image
        for o in range(self.batch_size):
            img = y_true[o]
            # find all part value on the image
            classes = np.unique(img)
            # remove background and 255
            classes = classes[1:]
            if 255 in classes:
                classes = classes[:-1]

            mat_contour = []
            # find for each class the contours
            for i in range(len(classes)):
                # current part
                value = classes[i]

                mask = cv2.inRange(img, int(value), int(value))
                # per ( perimeter) contains all found contours by  cv2.findContours for current part
                _, per, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

                # mat total will contain all perimeter points for current class
                mat_total = np.zeros(shape=(1, 2))
                # merge all these perimeters and point on a single array
                # loop on perimeters
                for q in range(len(per)):

                    tmp = per[q]
                    mat = np.zeros(shape=(len(tmp), 2))
                    # loop on perimeter points
                    for j in range(len(tmp)):
                        point = tmp[j]
                        x = point[0][0]
                        y = point[0][1]
                        mat[j][0] = x
                        mat[j][1] = y

                    mat_total = np.concatenate((mat_total, mat), axis=0)

                mat_contour.append(mat_total[1:])

            for i in range(len(classes)):
                # all contour points for class i
                tmp = mat_contour[i]
                # loop on i+1 list of points
                for j in range(i + 1, len(classes)):

                    min_v = sys.maxsize
                    second_mat = mat_contour[j]

                    # for each point on tmp
                    # calculate the distance all point of another countour class
                    for p in range(len(tmp)):
                        first_mat = tmp[p]

                        dif = first_mat - second_mat
                        dif = dif * dif
                        sum_mat = np.sum(dif, 1)
                        sqrt = np.sqrt(sum_mat)

                        min_tmp = np.min(sqrt)
                        if min_tmp < min_v:
                            min_v = min_tmp
                    # min_v è la distanza minima trovata tra tutti i punti di tmp e tutti i punti di second mat
                    # dove second mat è la lista dei punti di un altra classe
                    # se min_v è <= 1 allora le due classi sono adiacenti e pongo a 1 il valore sull adj_mat
                    if min_v <= self.pixel_distance:
                        adj_mat[classes[i]][classes[j]] = 1 + adj_mat[classes[i]][classes[j]]

        return adj_mat.astype(np.float32)


class WeightedAdjMat(object):
    """
    Weighted Adjacency matrix for all batch images
        count common-close pixel
    """

    def __init__(self, batch_size, pixel_distance=1):
        """
        :param batch_size:
        :param pixel_distance: looking on ground truth images some close
            and related parts are divided by some background pixel so they could not appear so close as they should
            pixel distance is used to weight in a more accurate way these cases
        """
        super().__init__()
        self.batch_size = batch_size
        self.pixel_distance = pixel_distance

    def adj_mat(self, y_true, y_pred):
        # Wraps np_mean_iou method and uses it as a TensorFlow op.
        # Takes numpy arrays as its arguments and returns numpy arrays as
        # its outputs.
        return tf.py_func(self.np_adj_func_2, [y_true, y_pred], tf.float32)

    # matrice pesata
    def np_adj_func_2(self, y_true, y_pred):

        adj_mat = np.zeros(shape=(108, 108))
        for o in range(self.batch_size):
            img = y_true[o]
            classes = np.unique(img)
            classes = classes[1:]
            if 255 in classes:
                classes = classes[:-1]
            mat_contour = []

            for i in range(len(classes)):

                value = classes[i]
                mask = cv2.inRange(img, int(value), int(value))
                _, per, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

                mat_total = np.zeros(shape=(1, 2))

                for q in range(len(per)):

                    tmp = per[q]
                    mat = np.zeros(shape=(len(tmp), 2))
                    for j in range(len(tmp)):
                        point = tmp[j]
                        x = point[0][0]
                        y = point[0][1]
                        mat[j][0] = x
                        mat[j][1] = y

                    mat_total = np.concatenate((mat_total, mat), axis=0)

                mat_contour.append(mat_total[1:])

            for i in range(len(classes)):
                tmp = mat_contour[i]

                for j in range(i + 1, len(classes)):

                    # min_v = sys.maxsize
                    second_mat = mat_contour[j]
                    adj_pixel = 0

                    for p in range(len(tmp)):
                        first_mat = tmp[p]

                        dif = first_mat - second_mat
                        dif = dif * dif
                        sum_mat = np.sum(dif, 1)
                        sqrt = np.sqrt(sum_mat)

                        # sqrt is a vector with all point distances
                        # mask contains 1 value only where these distances are <= pixel distance
                        mask = sqrt <= self.pixel_distance
                        tmp_pixel = np.sum(mask)
                        adj_pixel = tmp_pixel + adj_pixel

                    if adj_pixel > 0:
                        adj_mat[classes[i]][classes[j]] = adj_pixel + adj_mat[classes[i]][classes[j]]

        return adj_mat.astype(np.float32)


class SingleAdjMat(object):
    """
    Create a weighted adjacency matrix for a single image
    it differs from WeightedAdjMat because it does not loop on batch image
        but create a adj matrix only for the imput image
    """

    def __init__(self, batch_size, index, pixel_distance=1):
        super().__init__()
        self.batch_size = batch_size
        self.index = index
        self.pixel_distance = pixel_distance

    def adj_mat(self, y_true, y_pred):
        # Wraps np_mean_iou method and uses it as a TensorFlow op.
        # Takes numpy arrays as its arguments and returns numpy arrays as
        # its outputs.
        return tf.py_func(self.np_adj_func_4, [y_true, y_pred], tf.float32)

    def np_adj_func_4(self, y_true, y_pred):

        adj_mat = np.zeros(shape=(108, 108))

        img = y_true[self.index]
        classes = np.unique(img)
        classes = classes[1:]
        if 255 in classes:
            classes = classes[:-1]
        mat_contour = []

        for i in range(len(classes)):

            value = classes[i]
            mask = cv2.inRange(img, int(value), int(value))
            _, per, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

            mat_total = np.zeros(shape=(1, 2))

            for q in range(len(per)):

                tmp = per[q]
                mat = np.zeros(shape=(len(tmp), 2))
                for j in range(len(tmp)):
                    point = tmp[j]
                    x = point[0][0]
                    y = point[0][1]
                    mat[j][0] = x
                    mat[j][1] = y

                mat_total = np.concatenate((mat_total, mat), axis=0)

            mat_contour.append(mat_total[1:])

        for i in range(len(classes)):
            tmp = mat_contour[i]

            for j in range(i + 1, len(classes)):

                # min_v = sys.maxsize
                second_mat = mat_contour[j]
                adj_pixel = 0

                for p in range(len(tmp)):
                    first_mat = tmp[p]

                    dif = first_mat - second_mat
                    dif = dif * dif
                    sum_mat = np.sum(dif, 1)
                    sqrt = np.sqrt(sum_mat)

                    mask = sqrt <= self.pixel_distance
                    tmp_pixel = np.sum(mask)
                    adj_pixel = tmp_pixel + adj_pixel

                if adj_pixel > 0:
                    adj_mat[classes[i]][classes[j]] = adj_pixel + adj_mat[classes[i]][classes[j]]

        return adj_mat.astype(np.float32)


"""
la classe che segue era solo per fare alcuni test di prova
"""

class adj_mat_func(object):
    def __init__(self, batch_size, index):
        super().__init__()
        self.batch_size = batch_size
        self.index = 0

    def adj_mat(self, y_true, y_pred):
        # Wraps np_mean_iou method and uses it as a TensorFlow op.
        # Takes numpy arrays as its arguments and returns numpy arrays as
        # its outputs.
        return tf.py_func(self.np_adj_func, [y_true, y_pred], tf.float32)

    def np_adj_func(self, y_true, y_pred):

        adj_mat = np.zeros(shape=(108, 108))
        for o in range(self.batch_size):
            img = y_true[o]
            classes = np.unique(img)
            classes = classes[1:]
            if 255 in classes:
                classes = classes[:-1]
            mat_contour = []

            for i in range(len(classes)):

                value = classes[i]
                mask = cv2.inRange(img, int(value), int(value))
                _, per, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

                mat_total = np.zeros(shape=(1, 2))

                for q in range(len(per)):

                    tmp = per[q]
                    mat = np.zeros(shape=(len(tmp), 2))
                    for j in range(len(tmp)):
                        point = tmp[j]
                        x = point[0][0]
                        y = point[0][1]
                        mat[j][0] = x
                        mat[j][1] = y

                    mat_total = np.concatenate((mat_total, mat), axis=0)

                mat_contour.append(mat_total[1:])

            for i in range(len(classes)):
                tmp = mat_contour[i]

                for j in range(i + 1, len(classes)):

                    min_v = sys.maxsize
                    second_mat = mat_contour[j]

                    for p in range(len(tmp)):
                        first_mat = tmp[p]

                        dif = first_mat - second_mat
                        dif = dif * dif
                        sum_mat = np.sum(dif, 1)
                        sqrt = np.sqrt(sum_mat)

                        min_tmp = np.min(sqrt)
                        if min_tmp < min_v:
                            min_v = min_tmp

                    if min_v <= 1:
                        adj_mat[classes[i]][classes[j]] = 1 + adj_mat[classes[i]][classes[j]]

        return adj_mat.astype(np.float32)

    # matrice pesata
    def np_adj_func_2(self, y_true, y_pred):

        adj_mat = np.zeros(shape=(108, 108))
        for o in range(self.batch_size):
            img = y_true[o]
            classes = np.unique(img)
            classes = classes[1:]
            if 255 in classes:
                classes = classes[:-1]
            mat_contour = []

            for i in range(len(classes)):

                value = classes[i]
                mask = cv2.inRange(img, int(value), int(value))
                _, per, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

                mat_total = np.zeros(shape=(1, 2))

                for q in range(len(per)):

                    tmp = per[q]
                    mat = np.zeros(shape=(len(tmp), 2))
                    for j in range(len(tmp)):
                        point = tmp[j]
                        x = point[0][0]
                        y = point[0][1]
                        mat[j][0] = x
                        mat[j][1] = y

                    mat_total = np.concatenate((mat_total, mat), axis=0)

                mat_contour.append(mat_total[1:])

            for i in range(len(classes)):
                tmp = mat_contour[i]

                for j in range(i + 1, len(classes)):

                    # min_v = sys.maxsize
                    second_mat = mat_contour[j]
                    adj_pixel = 0

                    for p in range(len(tmp)):
                        first_mat = tmp[p]

                        dif = first_mat - second_mat
                        dif = dif * dif
                        sum_mat = np.sum(dif, 1)
                        sqrt = np.sqrt(sum_mat)

                        mask = sqrt <= 1
                        tmp_pixel = np.sum(mask)
                        adj_pixel = tmp_pixel + adj_pixel

                    if adj_pixel > 0:
                        adj_mat[classes[i]][classes[j]] = adj_pixel + adj_mat[classes[i]][classes[j]]

        return adj_mat.astype(np.float32)

    def np_adj_func_4(self, y_true, y_pred):

        adj_mat = np.zeros(shape=(108, 108))

        img = y_true[self.index]
        classes = np.unique(img)
        classes = classes[1:]
        if 255 in classes:
            classes = classes[:-1]
        mat_contour = []

        for i in range(len(classes)):

            value = classes[i]
            mask = cv2.inRange(img, int(value), int(value))
            _, per, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

            mat_total = np.zeros(shape=(1, 2))

            for q in range(len(per)):

                tmp = per[q]
                mat = np.zeros(shape=(len(tmp), 2))
                for j in range(len(tmp)):
                    point = tmp[j]
                    x = point[0][0]
                    y = point[0][1]
                    mat[j][0] = x
                    mat[j][1] = y

                mat_total = np.concatenate((mat_total, mat), axis=0)

            mat_contour.append(mat_total[1:])

        for i in range(len(classes)):
            tmp = mat_contour[i]

            for j in range(i + 1, len(classes)):

                # min_v = sys.maxsize
                second_mat = mat_contour[j]
                adj_pixel = 0

                for p in range(len(tmp)):
                    first_mat = tmp[p]

                    dif = first_mat - second_mat
                    dif = dif * dif
                    sum_mat = np.sum(dif, 1)
                    sqrt = np.sqrt(sum_mat)

                    mask = sqrt <= 1
                    tmp_pixel = np.sum(mask)
                    adj_pixel = tmp_pixel + adj_pixel

                if adj_pixel > 0:
                    adj_mat[classes[i]][classes[j]] = adj_pixel + adj_mat[classes[i]][classes[j]]

        return adj_mat.astype(np.float32)
