#!/usr/bin/python
# -*- enconding:utf-8 -*-
import tensorflow as tf
#参数概要(ready to use tensorboard)
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)  #avg
        with tf.name_scope('stddev'):
            stddev=tf.sqrt((tf.reduce_mean(tf.square(var-mean))))
        tf.summary.scalar('stddev',stddev)
        # tf.summary.scalar('max',tf.reduce_max(var))
        # tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)   #直方图
