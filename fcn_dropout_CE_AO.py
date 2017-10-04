#!-- encoding:utf-8 --
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.tensor_board import variable_summaries
import time
import os

def mnistdataclassical_dropout_CE_AO(mnistDataDir="MNIST_data",batchSize=200,learningRate=0.01,epochSize=50,dropout=0.7):
    # 载入数据
    mnist = input_data.read_data_sets(mnistDataDir, one_hot=True)

    # 每个批次的大小(观察下更改批次大小的效果)
    batch_size = batchSize
    # 计算一共有多少个批次
    n_batch = mnist.train.num_examples // batch_size

    with tf.name_scope('Input_layer'):
        x = tf.placeholder(tf.float32, [None, 784],name='x_input')
        y = tf.placeholder(tf.float32, [None, 10],name='y_input')
        keep_prob = tf.placeholder(tf.float32,name='dropout_input')
        lr=tf.Variable(learningRate,dtype=tf.float32,name='learning_rate')
	tf.summary.scalar('learning_rate',lr)

    with tf.name_scope('Layer_1'):
        with tf.name_scope('Weights_1'):
            # 尝试增加隐藏层、调整w、b的初始化、更换不同的激活函数、分类函数观察准确率的变化
            w1 = tf.Variable(tf.truncated_normal([784, 1000],stddev=0.1,name='w1'))
            variable_summaries(w1)
        with tf.name_scope('Biases_1'):
            b1 = tf.Variable(tf.zeros([1000])+0.1,name='b1')
            variable_summaries(b1)
        with tf.name_scope('L1_Tanh_dropout'):
            L1=tf.nn.tanh(tf.matmul(x,w1)+b1)
            L1_drop=tf.nn.dropout(L1,keep_prob)

    with tf.name_scope('Layer_2'):
        with tf.name_scope('Weights_2'):
            w2 = tf.Variable(tf.truncated_normal([1000, 100], stddev=0.1,name='w2'))
            variable_summaries(w2)
        with tf.name_scope('Biases_2'):
            b2 = tf.Variable(tf.zeros([100]) + 0.1,name='b2')
            variable_summaries(b2)
        with tf.name_scope('L2_Tanh_dropout'):
            L2 = tf.nn.tanh(tf.matmul(L1_drop, w2) + b2)
            L2_drop = tf.nn.dropout(L2, keep_prob)

    with tf.name_scope('Layer_output'):
        with tf.name_scope('Weights_output'):
            w = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1,name='w'))
            variable_summaries(w)
        with tf.name_scope('Biases_output'):
            b = tf.Variable(tf.zeros([10])+ 0.1,name='b')
            variable_summaries(b)
        with tf.name_scope('softmax'):
            prediction = tf.nn.softmax(tf.matmul(L2_drop, w) + b)

    with tf.name_scope('Loss'):
        # 观察下不同的loss function对准确率的影响，以及学习率大小对结果的影响
        # loss = tf.reduce_mean(tf.square(y - prediction))
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
        tf.summary.scalar('loss',loss)
    with tf.name_scope('Train'):
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.name_scope('Accuracy'):
        with tf.name_scope('Correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
        # 求准确率
        with tf.name_scope('Accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    # merge all summary
    merge=tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)
        nowtime=time.strftime('%m%d%H%M')
        if not os.path.exists('tensorboard/chart%s' % nowtime):
            os.makedirs('tensorboard/chart%s' % nowtime)

        writer=tf.summary.FileWriter('tensorboard/chart%s/'%nowtime,sess.graph)
        for epoch in range(epochSize):  # 所有图片训练epoch_size次，（尝试修改不同的训练次数观察影响结果）
            sess.run(tf.assign(lr,learningRate*(0.95**epoch)))
            for batch in range(n_batch):  # 迭代所有图片
                batch_xdata, batch_ylabel = mnist.train.next_batch(batch_size)
                summary,_=sess.run([merge,train_step], feed_dict={x: batch_xdata, y: batch_ylabel, keep_prob:dropout})

            writer.add_summary(summary,epoch)

            learning_rate=sess.run(lr)
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob:dropout})  # 用测试集的所有图片和所有标签来观察模型准确率
            print("Iter : " + str(epoch) + " , Testing Accuracy : " + str(test_acc)+" , learning Rate: "+str(learning_rate))

        with open('tensorboard/run%s.txt'%time.strftime('%m%d'), 'a') as f:
            f.write('--------------------dropout_CE AdamOptimizer chart%s-------------------\n'% nowtime)
            f.write('%s\n'%time.strftime('%x %X'))
            f.write("Batch: %d \n" % batchSize)
            f.write("Epoch: %d \n" % epochSize)
            f.write("Dropout: %f \n" % dropout)
            f.write("Learning Rate: %f \n" % sess.run(lr))
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob:dropout})
            f.write("Testing Accuracy: %f \n" % test_acc)
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob:dropout})
            f.write("Training Accuracy: %f \n" % train_acc)
            f.write('Loss_CE: %f \n\n' % (sess.run(loss, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob:dropout})))






