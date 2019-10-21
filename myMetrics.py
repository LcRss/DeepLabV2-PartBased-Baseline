from Utils import *
from adj_mat_func import adj_mat_func
from tensorflow.python.keras import backend as k


def custom_adj_loss_l1(batch_size, adj_train, lambda_loss=1):
    def adj_loss(y_true, y_pred):
        Y_pred = k.argmax(y_pred)
        Y_true = k.argmax(y_true)

        adj = adj_mat_func(batch_size)

        adj_pred = adj.adj_mat(Y_pred, Y_true)
        adj_pred = tf.norm(tensor=adj_pred, ord=1, axis=1)
        adj_true = adj.adj_mat(Y_true, Y_pred)
        adj_true = tf.norm(tensor=adj_true, ord=1, axis=1)

        # L1
        mod = k.abs(adj_pred - adj_true)
        global adj_loss_value
        adj_loss_value = lambda_loss * k.mean(mod)

        global categ_loss
        categ_loss = k.categorical_crossentropy(y_true, y_pred)

        loss = adj_loss_value + categ_loss

        return loss

    return adj_loss


def custom_adj_loss_l2(batch_size, adj_train, lambda_loss=1):
    def adj_loss(y_true, y_pred):
        Y_pred = k.argmax(y_pred)
        Y_true = k.argmax(y_true)

        adj = adj_mat_func(batch_size)

        adj_pred = adj.adj_mat(Y_pred, Y_true)
        adj_pred = tf.norm(tensor=adj_pred, ord=1, axis=1)
        adj_true = adj.adj_mat(Y_true, Y_pred)
        adj_true = tf.norm(tensor=adj_true, ord=1, axis=1)

        # L2
        quad = (adj_pred - adj_true)
        quad = quad * quad

        global adj_loss_value
        adj_loss_value = lambda_loss * k.mean(quad)

        global categ_loss
        categ_loss = k.categorical_crossentropy(y_true, y_pred)

        loss = adj_loss_value + categ_loss

        return loss

    return adj_loss


def custom_adj_loss_frobenius(batch_size, adj_train, lambda_loss=1):
    def adj_loss(y_true, y_pred):
        Y_pred = k.argmax(y_pred)
        Y_true = k.argmax(y_true)

        adj = adj_mat_func(batch_size)

        adj_pred = adj.adj_mat(Y_pred, Y_true)
        adj_pred = tf.norm(tensor=adj_pred, ord=1, axis=1)
        adj_true = adj.adj_mat(Y_true, Y_pred)
        adj_true = tf.norm(tensor=adj_true, ord=1, axis=1)

        # L2
        quad = (adj_pred - adj_true)
        quad = quad * quad
        sqrt = k.sqrt(quad)

        global adj_loss_value
        adj_loss_value = lambda_loss * k.mean(sqrt)

        global categ_loss
        categ_loss = k.categorical_crossentropy(y_true, y_pred)

        loss = adj_loss_value + categ_loss

        return loss

    return adj_loss


def metric_adj(y_true, y_pred):
    return adj_loss_value


def metric_categ_cross(y_true, y_pred):
    return categ_loss


def custom_loss(layer):
    def cat_loss(y_true, y_pred):
        logits = layer.output
        loss = k.categorical_crossentropy(y_true, logits, from_logits=True)
        mask = k.sum(y_true, -1)
        mask = mask > 0
        loss = tf.boolean_mask(loss, mask)
        loss = k.mean(loss, axis=None, keepdims=False)

        return loss

    # Return a function
    return cat_loss


def compute_and_print_IoU_per_class(confusion_matrix, num_classes, class_mask=None, namePart=[]):
    """
    Computes and prints mean intersection over union divided per class
    :param confusion_matrix: confusion matrix needed for the computation
    """
    mIoU = 0
    mIoU_nobackgroud = 0
    IoU_per_class = np.zeros([num_classes], np.float32)
    true_classes = 0

    per_class_pixel_acc = np.zeros([num_classes], np.float32)

    mean_class_acc_num = 0

    # out = ''
    # out_pixel_acc = ''
    # index = ''

    true_classes_pix = 0
    mean_class_acc_den = 0

    mean_class_acc_num_nobgr = 0
    mean_class_acc_den_nobgr = 0
    mean_class_acc_sum_nobgr = 0
    mean_class_acc_sum = 0

    if class_mask == None:
        class_mask = np.ones([num_classes], np.int8)

    for i in range(num_classes):

        if class_mask[i] == 1:
            # IoU = true_positive / (true_positive + false_positive + false_negative)
            TP = confusion_matrix[i, i]
            FP = np.sum(confusion_matrix[:, i]) - TP
            FN = np.sum(confusion_matrix[i]) - TP
            # TN = np.sum(confusion_matrix) - TP - FP - FN

            denominator = (TP + FP + FN)
            # If the denominator is 0, we need to ignore the class.
            if denominator == 0:
                denominator = 1
                print(namePart[i])
            else:
                true_classes += 1

            # per-class pixel accuracy
            if not TP == 0:
                # if not np.isnan(TP):
                tmp = (TP + FN)
                per_class_pixel_acc[i] = TP / tmp

            IoU = TP / denominator
            IoU_per_class[i] += IoU
            mIoU += IoU

            if i > 0:
                mIoU_nobackgroud += IoU

            # mean class accuracy
            if not np.isnan(per_class_pixel_acc[i]):
                mean_class_acc_num += TP
                mean_class_acc_den += TP + FN

                mean_class_acc_sum += per_class_pixel_acc[i]
                true_classes_pix += 1

                if i > 0:
                    mean_class_acc_num_nobgr += TP
                    mean_class_acc_den_nobgr += TP + FN
                    mean_class_acc_sum_nobgr += per_class_pixel_acc[i]

    mIoU = mIoU / true_classes
    mIoU_nobackgroud = mIoU_nobackgroud / (true_classes - 1)

    mean_pix_acc = mean_class_acc_num / mean_class_acc_den
    mean_pixel_acc_nobackground = mean_class_acc_num_nobgr / mean_class_acc_den_nobgr

    print("---------------------------------------------------------------------------")
    print("True_classes: " + str(true_classes))
    print("---------------------------------------------------------------------------")
    print("-- background --")
    print("IoU for class background : " + str(IoU_per_class[0] * 100))
    print("Pixel Acc for class background : " + str(per_class_pixel_acc[0] * 100))
    print("---------------------------------------------------------------------------")

    zero_classes = []

    for k in range(1, num_classes):
        if IoU_per_class[k] > 0:
            print("-- " + str(k) + " -- " + namePart[k] + " --")
            print("IoU for class " + namePart[k] + " :" + str(IoU_per_class[k] * 100))
            print("Pixel Acc for class " + str(k) + " :" + str(per_class_pixel_acc[k] * 100))
            print("---------------------------------------------------------------------------")
        else:
            zero_classes.append(k)

    print(" ")
    print("--METRICS--")
    print(' mean_class_acc :' + str((mean_class_acc_sum / true_classes_pix) * 100))
    print(' mean pix acc :' + str(mean_pix_acc * 100))
    print(' mean_pixel_acc_no_background :' + str(mean_pixel_acc_nobackground * 100))
    print(" mIoU:" + str(mIoU * 100))
    print(" mIoU_nobackgroud:" + str(mIoU_nobackgroud * 100))
    print("---------------------------------------------------------------------------")

    for j in range(len(zero_classes)):
        print("class not found " + str(zero_classes[j]) + "_" + namePart[zero_classes[j]])

    print("Classes_" + str(108 - len(zero_classes)) + "/108")

    return mIoU * 100, IoU_per_class

# def custom_adj_loss(batch, adj_train):
#     def adj_loss(y_true, y_pred):
#         # adj_mat[classes[j]][classes[i]] = 1 + adj_mat[classes[j]][classes[i]]
#
#         loss = k.variable(value=tf.reduce_mean(tf.abs(adj_mat)))
#
#         return loss
#
#     # Return a function
#     return adj_loss


# def custom_metric_adj_loss(batch_size, adj_train):
#     def adj_loss(y_true, y_pred):
#         Y_pred = k.argmax(y_pred)
#         Y_true = k.argmax(y_true)
#
#         adj = adj_mat_func(batch_size)
#
#         adj_pred = adj.adj_mat(Y_pred, Y_true)
#         adj_pred = tf.norm(tensor=adj_pred, ord=1, axis=1)
#
#         adj_true = adj.adj_mat(Y_true, Y_pred)
#         adj_true = tf.norm(tensor=adj_true, ord=1, axis=1)
#
#         mod = k.abs(adj_pred - adj_true)
#         loss = k.mean(mod)
#
#         return loss
#
#     return adj_loss


# def custom_categ_losses(y_true, y_pred):
#     loss = k.categorical_crossentropy(y_true, y_pred, from_logits=True)
#     mask = k.sum(y_true, -1)
#     mask = mask > 0
#     loss = tf.boolean_mask(loss, mask)
#     loss = k.mean(loss, axis=None, keepdims=False)
#
#     return loss

# from tensorflow.python.keras import metrics
# def accuracy_ignoring_first_label(y_true, y_pred):
#     Y_true = y_true[:, :, 1:]
#     Y_pred = y_pred[:, :, 1:]
#     return metrics.categorical_accuracy(Y_true, Y_pred)
