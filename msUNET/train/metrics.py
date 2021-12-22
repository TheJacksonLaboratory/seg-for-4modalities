# Metric function for Mouse Brain Segmentation in UNET
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


def dice_coef(y_true, y_pred):
    # Deprectated
    # Function that calculates the dice score between two binary input samples
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + K.epsilon()) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


def dice_coef_metric(y_true, y_pred):
    # Deprecated
    return dice_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    # Deprecated
    return -dice_coef(y_true, y_pred) + 1


class dice_coef_loss_keras(tf.keras.losses.Loss):
    # Deprecated
    # Class that calculates the dice coefficient loss between two binary input samples
    # Inherits from keras custom loss class for robustness
    def call(self, y_true, y_pred):
        print(y_true.shape)
        print(y_pred.shape)
        print(-dice_coef(y_true, y_pred) + 1)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        loss_val = 1 - ((2.0 * intersection + K.epsilon()) /
                        (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon()))
        ones_array = tf.constant(1, shape=[128], dtype=tf.float32)
        loss = tf.math.scalar_mul(loss_val, ones_array)
        return loss


class dice_coef_loss_by_sample(tf.keras.losses.Loss):
    # Class that calculates dice coeffieicnt between two binary input samples
    # Unlike previous iterations, calculates dice coeffieicnt on a by-sample basis
    # instead of on a by-slice basis. This allows for the addition of variables weights.
    # In our case, we use that capability to weight different data modalities
    # differently
    def call(self, y_true, y_pred):
        y_pred_f = K.reshape(y_pred, [-1, y_pred.shape[1] * y_pred.shape[2]])
        y_true_f = K.reshape(y_true, [-1, y_pred.shape[1] * y_pred.shape[2]])
        print(y_pred_f.shape)
        intersection = K.sum(y_true_f * y_pred_f, axis=1)
        print(intersection.shape)
        loss_num = tf.math.add(
            tf.math.scalar_mul(
                2, intersection), K.epsilon())
        print(loss_num.shape)
        loss_den = tf.math.add(
            tf.math.add(
                K.sum(
                    y_true_f, axis=1), K.sum(
                    y_pred_f, axis=1)), K.epsilon())
        print(loss_den.shape)
        dice = tf.math.divide(loss_num, loss_den)
        print(dice.shape)
        loss = tf.math.add(1.0, tf.math.scalar_mul(-1, dice))
        print(loss.shape)

        return loss


def jaccard_similarity(y_true, y_pred):
    # Function that calculates the jaccard similarity between two binary
    # samples
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ((intersection + K.epsilon())
            / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + K.epsilon()))


def jaccard_distance(y_true, y_pred):
    return 1 - jaccard_similarity(y_true, y_pred)
