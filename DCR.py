import tensorflow as tf
import vgg16
import cv2
import numpy as np

import multiprocessing
import os

import tensorflow.contrib.slim as slim
from config import cfg

#import sys
#import importlib
#importlib.reload(sys)

img_size = 352
label_size = img_size
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class Model:
    def __init__(self):
        self.vgg = vgg16.Vgg16()

        self.input_holder = tf.placeholder(tf.float32, [1, img_size, img_size, 3])
        self.label_holder = tf.placeholder(tf.float32, [label_size*label_size, 2])

        self.sobel_fx, self.sobel_fy = self.sobel_filter()

        self.contour_th = 1.5
        self.contour_weight = 0.0001

    def build_model(self):

        #build the VGG-16 model
        vgg = self.vgg
        vgg.build(self.input_holder)

        fea_dim = 128
        data_size = 352
        dim_trans = 1
        weights_regularizer = tf.contrib.layers.l2_regularizer(5e-04)
        
        
        # Dilation
        self.Fea_P1 = self.dilation(vgg.conv1_2, 64, fea_dim/4, 'Fea_P1')
        self.Fea_P2 = self.dilation(vgg.conv2_2, 128, fea_dim/4, 'Fea_P2')
        self.Fea_P3 = self.dilation(vgg.conv3_3, 256, fea_dim/4, 'Fea_P3')
        self.Fea_P4 = self.dilation(vgg.conv4_3, 512, fea_dim/4, 'Fea_P4')
        self.Fea_P5 = self.dilation(vgg.conv5_3, 512, fea_dim/4, 'Fea_P5')	
        
        #self.Fea_P5_Up = tf.nn.relu(self.Deconv_2d(self.Fea_P5, [1, int(np.floor(data_size / 8)), int(np.floor(data_size / 8)), fea_dim], 5, 2, name='Fea_P5_Deconv'))
        #self.Fea_P4_Concat = self.Conv_2d(tf.concat([self.Fea_P4, self.Fea_P5_Up], axis=3), [1, 1, fea_dim*2, fea_dim], 0.01, padding='VALID', name='Fea_P4_Concat')
    
        #self.Fea_P4_Concat_Up = tf.nn.relu(self.Deconv_2d(self.Fea_P4_Concat, [1, int(np.floor(data_size / 4)), int(np.floor(data_size / 4)), fea_dim], 5, 2, name='Fea_P4_Concat_Deconv'))
        #self.Fea_P3_Concat = self.Conv_2d(tf.concat([self.Fea_P3, self.Fea_P4_Concat_Up], axis=3), [1, 1, fea_dim*2, fea_dim], 0.01, padding='VALID', name='Fea_P3_Concat')  
    
        #self.Fea_P3_Concat_Up = tf.nn.relu(self.Deconv_2d(self.Fea_P3_Concat, [1, int(np.floor(data_size / 2)), int(np.floor(data_size / 2)), fea_dim], 5, 2, name='Fea_P2_Concat_Deconv'))
        #self.Fea_P2_Concat = self.Conv_2d(tf.concat([self.Fea_P2, self.Fea_P3_Concat_Up], axis=3), [1, 1, fea_dim*2, fea_dim], 0.01, padding='VALID', name='Fea_P2_Concat')
    
        #self.Fea_P2_Concat_Up = tf.nn.relu(self.Deconv_2d(self.Fea_P2_Concat, [1, data_size, data_size, fea_dim], 5, 2, name='Fea_P1_Concat_Deconv'))
        #self.Fea_P1_Concat = tf.nn.relu(self.Conv_2d(tf.concat([self.Fea_P1, self.Fea_P2_Concat_Up], axis=3), [1, 1, fea_dim*2, fea_dim], 0.01, padding='VALID', name='Fea_P1_Concat') )        
        
        
    
        #capsule attention
        with tf.variable_scope('relu_conv5') as scope:
            self.Fea5 = slim.conv2d(self.Fea_P5, num_outputs=fea_dim, kernel_size=[
                                 3, 3], stride=1, padding='SAME', scope='conv1', activation_fn=tf.nn.relu)
        
            data_size = int(np.floor(data_size / 16)) 
        
            self.output = self.primary(self.Fea5, 1, cfg.B, data_size, 5)
        
            self.output = self.convcaps1(self.output, 1, data_size, cfg.B, cfg.C, dim_trans, weights_regularizer, 5)
        
        self.capsg5 = self.Conv_2d(tf.concat([self.output, self.Fea5], axis=3), [1, 1, cfg.C * 17 + fea_dim, fea_dim], 0.01, padding='VALID', name='Fea_P5_Concat')
        self.capsg5 = self.capsg5 + self.Fea5        
        #data_size = int(np.floor(data_size / 2))
        
        data_size = data_size * 2
        self.capsg5_UP = tf.nn.relu(self.Deconv_2d(self.capsg5, [1, data_size, data_size, fea_dim], 5, 2, name='Capsg5_Deconv'))
        self.Fea4 = self.Conv_2d(tf.concat([self.capsg5_UP, self.Fea_P4], axis=3), [1, 1, fea_dim + fea_dim, fea_dim], 0.01, padding='VALID', name='Fea4')
        
        with tf.variable_scope('relu_conv4') as scope:
            self.Fea4 = slim.conv2d(self.Fea4, num_outputs=fea_dim, kernel_size=[
                                 3, 3], stride=1, padding='SAME', scope='conv1', activation_fn=tf.nn.relu)
            
        #data_size = int(np.floor(data_size / 16)) 
        
            self.output = self.primary(self.Fea4, 1, cfg.B, data_size, 5)
        
            self.output = self.convcaps1(self.output, 1, data_size, cfg.B, cfg.C, dim_trans, weights_regularizer, 5)
        
            self.capsg4 = self.Conv_2d(tf.concat([self.output, self.Fea4], axis=3), [1, 1, cfg.C * 17 + fea_dim, fea_dim], 0.01, padding='VALID', name='Fea_P4_Concat')
            self.capsg4 = self.capsg4 + self.Fea4
        
        data_size = data_size * 2
        self.capsg4_UP = tf.nn.relu(self.Deconv_2d(self.capsg4, [1, data_size, data_size, fea_dim], 5, 2, name='Capsg4_Deconv'))
        self.Fea3 = self.Conv_2d(tf.concat([self.capsg4_UP, self.Fea_P3], axis=3), [1, 1, fea_dim + fea_dim, fea_dim], 0.01, padding='VALID', name='Fea3')
        
        with tf.variable_scope('relu_conv3') as scope:
            self.Fea3 = slim.conv2d(self.Fea3, num_outputs=fea_dim, kernel_size=[
                                 3, 3], stride=1, padding='SAME', scope='conv1', activation_fn=tf.nn.relu)
        
        #data_size = int(np.floor(data_size / 16)) 
        
            self.output = self.primary(self.Fea3, 1, cfg.B, data_size, 5)
        
            self.output = self.convcaps1(self.output, 1, data_size, cfg.B, cfg.C, dim_trans, weights_regularizer, 5) 
        
        self.capsg3 = self.Conv_2d(tf.concat([self.output, self.Fea3], axis=3), [1, 1, cfg.C * 17 + fea_dim, fea_dim], 0.01, padding='VALID', name='Fea_P3_Concat')
        self.capsg3 = self.capsg3 + self.Fea3
        
        data_size = data_size * 2
        self.capsg3_UP = tf.nn.relu(self.Deconv_2d(self.capsg3, [1, data_size, data_size, fea_dim], 5, 2, name='Capsg3_Deconv'))
        self.Fea2 = self.Conv_2d(tf.concat([self.capsg3_UP, self.Fea_P2], axis=3), [1, 1, fea_dim + fea_dim, fea_dim], 0.01, padding='VALID', name='Fea2')
        
        #with tf.variable_scope('relu_conv2') as scope:
            #self.Fea2 = slim.conv2d(self.Fea2, num_outputs=fea_dim, kernel_size=[
                                 #3, 3], stride=1, padding='SAME', scope='conv1', activation_fn=tf.nn.relu)
        
        ##data_size = int(np.floor(data_size / 16)) 
        
            #self.output = self.primary(self.Fea2, 1, cfg.B, data_size, 5)
        
            #self.output = self.convcaps1(self.output, 1, data_size, cfg.B, cfg.C, dim_trans, weights_regularizer, 5)
        
        #self.capsg2 = self.Conv_2d(tf.concat([self.output, self.Fea2], axis=3), [1, 1, cfg.C * 17 + fea_dim, fea_dim], 0.01, padding='VALID', name='Fea_P2_Concat')
        #self.capsg2 = self.capsg2 + self.Fea2
        
        data_size = data_size * 2
        self.capsg2_UP = tf.nn.relu(self.Deconv_2d(self.Fea2, [1, data_size, data_size, fea_dim], 5, 2, name='Capsg2_Deconv'))
        self.Fea1 = self.Conv_2d(tf.concat([self.capsg2_UP, self.Fea_P1], axis=3), [1, 1, fea_dim + fea_dim, fea_dim], 0.01, padding='VALID', name='Fea1')
        
        #with tf.variable_scope('relu_conv1') as scope:
            #self.Fea1 = slim.conv2d(self.Fea1, num_outputs=fea_dim, kernel_size=[
                                 #3, 3], stride=1, padding='SAME', scope='conv1', activation_fn=tf.nn.relu)
        
        ##data_size = int(np.floor(data_size / 16)) 
        
            #self.output = self.primary(self.Fea1, 1, cfg.B, data_size, 5)
        
            #self.output = self.convcaps1(self.output, 1, data_size, cfg.B, cfg.C, dim_trans, weights_regularizer, 5)
        
        #self.capsg1 = self.Conv_2d(tf.concat([self.output, self.Fea1], axis=3), [1, 1, cfg.C * 17 + fea_dim, fea_dim], 0.01, padding='VALID', name='Fea_P1_Concat')
        #self.capsg1 = self.capsg1 + self.Fea1
        
        #data_size = data_size * 2
        #self.capsg1_UP = tf.nn.relu(self.Deconv_2d(self.capsg1, [1, data_size, data_size, fea_dim], 5, 2, name='Capsg1_Deconv'))
        
        #guidance
        #self.caps1 = tf.image.resize_images(self.caps_Score, [data_size, data_size])        
        #self.Concat1 = self.Conv_2d(tf.concat([self.Fea_P1_Concat, self.caps1], axis=3), [1, 1, fea_dim + 2, fea_dim], 0.01, padding='VALID', name='Concat1')
    
        #self.caps2 = tf.image.resize_images(self.caps_Score, [int(np.floor(data_size / 2)), int(np.floor(data_size / 2))])        
        #self.Concat2 = self.Conv_2d(tf.concat([self.Fea_P2_Concat, self.caps2], axis=3), [1, 1, fea_dim + 2, fea_dim], 0.01, padding='VALID', name='Concat2')
        
        #self.caps3 = tf.image.resize_images(self.caps_Score, [int(np.floor(data_size / 4)), int(np.floor(data_size / 4))])        
        #self.Concat3 = self.Conv_2d(tf.concat([self.Fea_P3_Concat, self.caps3], axis=3), [1, 1, fea_dim + 2, fea_dim], 0.01, padding='VALID', name='Concat3')
        
        #self.caps4 = tf.image.resize_images(self.caps_Score, [int(np.floor(data_size / 8)), int(np.floor(data_size / 8))])        
        #self.Concat4 = self.Conv_2d(tf.concat([self.Fea_P4_Concat, self.caps4], axis=3), [1, 1, fea_dim + 2, fea_dim], 0.01, padding='VALID', name='Concat4')
        
        #self.caps5 = tf.image.resize_images(self.caps_Score, [int(np.floor(data_size / 16)), int(np.floor(data_size / 16))])        
        #self.Concat5 = self.Conv_2d(tf.concat([self.Fea_P5, self.caps5], axis=3), [1, 1, fea_dim + 2, fea_dim], 0.01, padding='VALID', name='Concat5')
                
        
            
        #self.output_Score1 = self.Conv_2d(self.Concat1, [1, 1, fea_dim, 2], 0.01, padding='VALID', name='output_Score1')   
        
        #self.Concat2_Up = tf.nn.relu(self.Deconv_2d(self.Concat2, [1, data_size, data_size, fea_dim], 5, 2, name='Concat2Deconv'))
        #self.output_Score2 = self.Conv_2d(self.Concat2_Up, [1, 1, fea_dim, 2], 0.01, padding='VALID', name='output_Score2')   
        
        #self.Concat3_Up1 = tf.nn.relu(self.Deconv_2d(self.Concat3, [1, int(np.floor(data_size / 2)), int(np.floor(data_size / 2)), fea_dim], 5, 2, name='Concat3Deconv1'))
        #self.Concat3_Up2 = tf.nn.relu(self.Deconv_2d(self.Concat3_Up1, [1, data_size, data_size, fea_dim], 5, 2, name='Concat3Deconv2'))
        #self.output_Score3 = self.Conv_2d(self.Concat3_Up2, [1, 1, fea_dim, 2], 0.01, padding='VALID', name='output_Score3')
        
        #self.Concat4_Up1 = tf.nn.relu(self.Deconv_2d(self.Concat4, [1, int(np.floor(data_size / 4)), int(np.floor(data_size / 4)), fea_dim], 5, 2, name='Concat4Deconv1'))
        #self.Concat4_Up2 = tf.nn.relu(self.Deconv_2d(self.Concat4_Up1, [1, int(np.floor(data_size / 2)), int(np.floor(data_size / 2)), fea_dim], 5, 2, name='Concat4Deconv2'))
        #self.Concat4_Up3 = tf.nn.relu(self.Deconv_2d(self.Concat4_Up2, [1, data_size, data_size, fea_dim], 5, 2, name='Concat4Deconv3'))
        #self.output_Score4 = self.Conv_2d(self.Concat4_Up3, [1, 1, fea_dim, 2], 0.01, padding='VALID', name='output_Score4')
        
        #self.Concat5_Up1 = tf.nn.relu(self.Deconv_2d(self.Concat5, [1, int(np.floor(data_size / 8)), int(np.floor(data_size / 8)), fea_dim], 5, 2, name='Concat5Deconv1'))
        #self.Concat5_Up2 = tf.nn.relu(self.Deconv_2d(self.Concat5_Up1, [1, int(np.floor(data_size / 4)), int(np.floor(data_size / 4)), fea_dim], 5, 2, name='Concat5Deconv2'))
        #self.Concat5_Up3 = tf.nn.relu(self.Deconv_2d(self.Concat5_Up2, [1, int(np.floor(data_size / 2)), int(np.floor(data_size / 2)), fea_dim], 5, 2, name='Concat5Deconv3'))
        #self.Concat5_Up4 = tf.nn.relu(self.Deconv_2d(self.Concat5_Up3, [1, data_size, data_size, fea_dim], 5, 2, name='Concat5Deconv4'))
        self.output_Score = self.Conv_2d(self.Fea1, [1, 1, fea_dim, 2], 0.01, padding='VALID', name='output_Score')
        
        #self.output_Score = self.output_Score1 + self.output_Score1 +self.output_Score3 + self.output_Score4 + self.output_Score5 + self.caps1

        self.Score = tf.reshape(self.output_Score, [-1,2])

        self.Prob = tf.clip_by_value(tf.nn.softmax(self.Score), 1e-8, 1.0)
        #self.Prob = tf.nn.softmax(self.Score)

        #Get the contour term
        self.Prob_C = tf.reshape(self.Prob, [1, img_size, img_size, 2])
        #self.Prob_Grad = tf.tanh(self.im_gradient(self.Prob_C))
        self.Prob_Grad = tf.tanh(tf.reduce_sum(self.im_gradient(self.Prob_C), reduction_indices=3, keep_dims=True))

        self.label_C = tf.reshape(self.label_holder, [1, img_size, img_size, 2])
        self.label_Grad = tf.cast(tf.greater(self.im_gradient(self.label_C), self.contour_th), tf.float32)
        self.label_Grad = tf.cast(tf.greater(tf.reduce_sum(self.im_gradient(self.label_C),
                                                           reduction_indices=3, keep_dims=True),
                                             self.contour_th), tf.float32)

        self.C_IoU_LOSS = self.Loss_IoU(self.Prob_Grad, self.label_Grad)

        #self.Contour_Loss = self.Loss_Contour(self.Prob_Grad, self.label_Grad)

        #Loss Function
        self.Loss_Mean = self.C_IoU_LOSS \
                         + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Score,
                                                                                  labels=self.label_holder))
                                                                                  
        #self.Loss_Mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Score,
                                                                                  #labels=self.label_holder))

        self.correct_prediction = tf.equal(tf.argmax(self.Score,1), tf.argmax(self.label_holder, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def Conv_2d(self, input_, shape, stddev, name, padding='SAME'):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=shape,
                                initializer=tf.truncated_normal_initializer(stddev=stddev))

            conv = tf.nn.conv2d(input_, W, [1, 1, 1, 1], padding=padding)

            b = tf.Variable(tf.constant(0.0, shape=[shape[3]]), name='b')
            conv = tf.nn.bias_add(conv, b)

            return conv
        
    def Conv_2d_re(self, input_, shape, stddev, regularizer, name, padding='SAME'):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=shape,
                                initializer=tf.truncated_normal_initializer(stddev=stddev),
                                regularizer=regularizer)

            conv = tf.nn.conv2d(input_, W, [1, 1, 1, 1], padding=padding)

            b = tf.Variable(tf.constant(0.0, shape=[shape[3]]), name='b')
            conv = tf.nn.bias_add(conv, b)

            return conv 

    def Deconv_2d(self, input_, output_shape,
                  k_s=3, st_s=2, stddev=0.01, padding='SAME', name="deconv2d"):
        with tf.variable_scope(name):
            W = tf.get_variable('W',
                                shape=[k_s, k_s, output_shape[3], input_.get_shape()[3]],
                                initializer=tf.random_normal_initializer(stddev=stddev))

            deconv = tf.nn.conv2d_transpose(input_, W, output_shape=output_shape,
                                            strides=[1, st_s, st_s, 1], padding=padding)

            b = tf.get_variable('b', [output_shape[3]], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, b)

        return deconv
    
    def capsule_feature(self, input, data_size, dim, cfg_num):
        with tf.variable_scope('relu_conv1'+str(data_size)) as scope:
            output = slim.conv2d(input, num_outputs=dim, kernel_size=[
                                 3, 3], stride=1, padding='SAME', scope='deconv', activation_fn=tf.nn.relu)  
            #output = slim.conv2d(output, num_outputs=cfg_num1, kernel_size=[
                        #3, 3], stride=2, padding='SAME', scope=scope, activation_fn=tf.nn.relu)
            #data_size = int(np.floor((data_size - 4) / 2))
 
        with tf.variable_scope('primary_caps'+str(data_size)) as scope:
            pose = slim.conv2d(output, num_outputs=cfg_num * 16,
                               kernel_size=[1, 1], stride=1, padding='VALID', scope=scope, activation_fn=None)
            activation = slim.conv2d(output, num_outputs=cfg_num, kernel_size=[
                                     1, 1], stride=1, padding='VALID', scope='activation', activation_fn=tf.nn.sigmoid)
            pose = tf.reshape(pose, shape=[cfg.batch_size, data_size, data_size, cfg_num, 16])
            activation = tf.reshape(
                activation, shape=[cfg.batch_size, data_size, data_size, cfg_num, 1])
            output = tf.concat([pose, activation], axis=4) 
        
        return output
    
    
    def activation_trans_column(self, activation_input, data_size, cfg_num, dim, regularizer, s):
        activation = tf.reshape(activation_input, shape=[data_size, data_size, 3 * 3 * cfg_num, 1])
        activation = tf.transpose(activation, [3, 2, 0, 1])
        #activation = tf.nn.dropout(activation, 0.5)
        activation = self.Conv_2d_re(activation, [1, 1, data_size, dim], 0.01, regularizer, padding='VALID', name='activation_column' + str(s))
        activation = tf.transpose(activation, [2, 3, 1, 0])
        activation = tf.reshape(activation, shape=[dim*data_size, 3 * 3 * cfg_num, 1])
        activation = tf.nn.sigmoid(activation)
        
        return activation
    
    def pose_trans_column(self, pose_input, data_size, cfg_num, dim, regularizer, s):
        pose = tf.reshape(pose_input, shape=[data_size, data_size, 3 * 3 * cfg_num, 16])
        pose = tf.transpose(pose, [3, 2, 0, 1])
        #pose = self.Conv_2d_re(pose, [1, 1, data_size, 1], 0.01, regularizer, padding='VALID', name='pose_column')
        #pose = tf.nn.dropout(pose, 0.5)
        pose = self.Conv_2d_re(pose, [1, 1, data_size, dim], 0.01, regularizer, padding='VALID', name='pose_column' + str(s))
        pose = tf.transpose(pose, [2, 3, 1, 0])
        pose = tf.reshape(pose, shape=[dim*data_size, 3 * 3 * cfg_num, 16])
        
        return pose    
    
    def activation_trans_row(self, activation_input, data_size, cfg_num, dim, regularizer, s):
        activation = tf.reshape(activation_input, shape=[data_size, data_size, 3 * 3 * cfg_num, 1])
        activation = tf.transpose(activation, [3, 2, 1, 0])
        #activation = tf.nn.dropout(activation, 0.5)
        activation = self.Conv_2d_re(activation, [1, 1, data_size, dim], 0.01, regularizer, padding='VALID', name='activation_row' + str(s))
        activation = tf.transpose(activation, [3, 2, 1, 0])
        activation = tf.reshape(activation, shape=[dim*data_size, 3 * 3 * cfg_num, 1])
        activation = tf.nn.sigmoid(activation)
        
        return activation   
    
    def pose_trans_row(self, pose_input, data_size, cfg_num, dim, regularizer, s):
        pose = tf.reshape(pose_input, shape=[data_size, data_size, 3 * 3 * cfg_num, 16])
        pose = tf.transpose(pose, [3, 2, 1, 0])
        #pose = tf.nn.dropout(pose, 0.5)
        pose = self.Conv_2d_re(pose, [1, 1, data_size, dim], 0.01, regularizer, padding='VALID', name='pose_row' + str(s))
        pose = tf.transpose(pose, [3, 2, 1, 0])
        pose = tf.reshape(pose, shape=[dim*data_size, 3 * 3 * cfg_num, 16])
        
        return pose    
    
    def votes_trans_column(self, votes_input, data_size, cfg_num1, cfg_num2, regularizer):
        votes = tf.reshape(votes_input, shape=[data_size, data_size, 3 * 3 * cfg_num1, int(np.floor(cfg_num2/2))*16])
        votes = tf.transpose(votes, [3, 1, 2, 0])
        votes = self.Conv_2d_re(votes, [1, 1, data_size, 1], 0.01, regularizer, padding='VALID', name='votes_column')
        votes = tf.transpose(votes, [3, 1, 2, 0])
        votes = tf.reshape(votes, shape=[1*data_size, 3 * 3 * cfg_num1, cfg_num2, 16])   
        
        return votes
    
    def votes_trans_row(self, votes_input, data_size, cfg_num1, cfg_num2, regularizer):
        votes = tf.reshape(votes_input, shape=[data_size, data_size, 3 * 3 * cfg_num1, cfg_num2*16])
        votes = tf.transpose(votes, [0, 3, 2, 1])
        votes = self.Conv_2d_re(votes, [1, 1, data_size, 1], 0.01, regularizer, padding='VALID', name='votes_row')
        votes = tf.transpose(votes, [0, 3, 2, 1])
        votes = tf.reshape(votes, shape=[1*data_size, 3 * 3 * cfg_num1, cfg_num2, 16])   
        
        return votes    
    
    def miu_routing_trans_column(self, miu_input, data_size, cfg_num, dim, regularizer):
        miu = tf.reshape(miu_input, shape=[dim, data_size, cfg_num, 16])
        miu = tf.transpose(miu, [3, 1, 2, 0])
        miu = self.Conv_2d_re(miu, [1, 1, dim, data_size], 0.01, regularizer, padding='VALID', name='miu_routing_column')
        miu = tf.transpose(miu, [3, 1, 2, 0])        
        
        return miu
    
    def miu_routing_trans_row(self, miu_input, data_size, cfg_num, dim, regularizer):
        miu = tf.reshape(miu_input, shape=[data_size, dim, cfg_num, 16])
        miu = tf.transpose(miu, [0, 3, 2, 1])
        miu = self.Conv_2d_re(miu, [1, 1, dim, data_size], 0.01, regularizer, padding='VALID', name='miu_routing_row')
        miu = tf.transpose(miu, [0, 3, 2, 1])        
        
        return miu    
    
    def activation_routing_trans_column(self, activation_input, data_size, cfg_num, dim, regularizer):
        activation = tf.reshape(activation_input, shape=[dim, data_size, cfg_num, 1])
        activation = tf.transpose(activation, [3, 1, 2, 0])
        activation = self.Conv_2d_re(activation, [1, 1, dim, data_size], 0.01, regularizer, padding='VALID', name='activation_routing_column')
        activation = tf.transpose(activation, [3, 1, 2, 0]) 
        activation = tf.reshape(activation, shape=[data_size*data_size, cfg_num])   
        
        return activation
    
    def activation_routing_trans_row(self, activation_input, data_size, cfg_num, dim, regularizer):
        activation = tf.reshape(activation_input, shape=[data_size, dim, cfg_num, 1])
        activation = tf.transpose(activation, [0, 3, 2, 1])
        activation = self.Conv_2d_re(activation, [1, 1, dim, data_size], 0.01, regularizer, padding='VALID', name='activation_routing_row')
        activation = tf.transpose(activation, [0, 3, 2, 1]) 
        activation = tf.reshape(activation, shape=[data_size*data_size, cfg_num])
        
        return activation
    
    def miu_routing_trans(self, input1, input2, cfg_batch_size, data_size, dim, cfg_num, cfg_scope):
        input1 = tf.reshape(input1, [data_size, dim, cfg_num, 16])
        input2 = tf.reshape(input2, [dim, data_size, cfg_num, 16])
        input1_t = tf.transpose(input1, [2, 3, 0, 1])
        input2_t = tf.transpose(input2, [2, 3, 0, 1])
        miu = tf.matmul(input1_t, input2_t)
        miu = tf.transpose(miu, [2, 3, 0, 1])
        #miu = tf.reshape(tf.transpose(miu, [2, 3, 0, 1]), [data_size, data_size, -1])
        #miu = tf.reshape(miu, [cfg_batch_size, data_size, data_size, cfg_num*16])
        #with tf.variable_scope('miu'+str(cfg_scope)) as scope:
            #miu = slim.conv2d(miu, num_outputs=cfg_num * 16, kernel_size=[1, 1], stride=1, padding='VALID', scope=scope, activation_fn=None)
            ##miu = tf.nn.dropout(miu, 0.5)
            #miu = tf.reshape(miu, shape=[cfg_batch_size, data_size, data_size, cfg_num, 16])  
            #miu = tf.reshape(miu, shape=[data_size, data_size, cfg_num, 16])

        return miu
    
    def activation_routing_trans(self, input1, input2, cfg_batch_size, data_size, dim, cfg_num, cfg_scope):
        input1_t = tf.reshape(input1, shape=[data_size, dim, cfg_num, 1])
        input2_t = tf.reshape(input2, shape=[dim, data_size, cfg_num, 1])
        input1_t = tf.transpose(input1_t, [2, 3, 0, 1])
        input2_t = tf.transpose(input2_t, [2, 3, 0, 1])
        activation = tf.matmul(input1_t, input2_t)
        activation = tf.transpose(activation, [2, 3, 0, 1])
        activation = tf.reshape(activation, shape=[data_size*data_size, cfg_num*1])
        activation = tf.nn.sigmoid(activation)
        #activation = tf.reshape(tf.transpose(activation, [2, 3, 0, 1]), [data_size, data_size, -1])
        #activation = tf.reshape(activation, [cfg_batch_size, data_size, data_size, cfg_num*1])
        #with tf.variable_scope('activation'+str(cfg_scope)) as scope:
            #activation = slim.conv2d(activation, num_outputs=cfg_num, kernel_size=[1, 1], stride=1, padding='VALID', scope=scope, activation_fn=tf.nn.sigmoid)
            ##activation = tf.nn.dropout(activation, 0.5)
            #activation = tf.reshape(activation, shape=[cfg_batch_size, data_size, data_size, cfg_num, 1])        
            #activation = tf.reshape(activation, shape=[cfg_batch_size*data_size*data_size, cfg_num*1])
        
        return activation

    def Contrast_Layer(self, input_, k_s=3):
        h_s = int(k_s / 2)
        return tf.subtract(input_, tf.nn.avg_pool(tf.pad(input_, [[0, 0], [h_s, h_s], [h_s, h_s], [0, 0]], 'SYMMETRIC'),
                                                  ksize=[1, k_s, k_s, 1], strides=[1, 1, 1, 1], padding='VALID'))

    def sobel_filter(self):
        fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
        fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)

        fx = np.stack((fx, fx), axis=2)
        fy = np.stack((fy, fy), axis=2)

        fx = np.reshape(fx, (3, 3, 2, 1))
        fy = np.reshape(fy, (3, 3, 2, 1))

        tf_fx = tf.Variable(tf.constant(fx))
        tf_fy = tf.Variable(tf.constant(fy))

        return tf_fx, tf_fy

    def im_gradient(self, im):
        gx = tf.nn.depthwise_conv2d(tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC'),
                                    self.sobel_fx, [1, 1, 1, 1], padding='VALID')
        gy = tf.nn.depthwise_conv2d(tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC'),
                                    self.sobel_fy, [1, 1, 1, 1], padding='VALID')
        return tf.sqrt(tf.add(tf.square(gx), tf.square(gy)))

    def Loss_IoU(self, pred, gt):
        inter = tf.reduce_sum(tf.multiply(pred, gt))
        union = tf.add(tf.reduce_sum(tf.square(pred)), tf.reduce_sum(tf.square(gt)))

        if inter == 0:
            return 0
        else:
            return 1 - (2*(inter+1)/(union + 1))

    def Loss_Contour(self, pred, gt):
        return tf.reduce_mean(-gt*tf.log(pred+0.00001) - (1-gt)*tf.log(1-pred+0.00001))

    def L2(self, tensor, wd=0.0005):
        return tf.mul(tf.nn.l2_loss(tensor), wd, name='L2-Loss')
    
    def kernel_tile(self, input, kernel, stride):
        # output = tf.extract_image_patches(input, ksizes=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='VALID')
    
        input_shape = input.get_shape()
        tile_filter = np.zeros(shape=[kernel, kernel, input_shape[3],
                                      kernel * kernel], dtype=np.float32)
        for i in range(kernel):
            for j in range(kernel):
                tile_filter[i, j, :, i * kernel + j] = 1.0
    
        tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
        output = tf.nn.depthwise_conv2d(input, tile_filter_op, strides=[
                                        1, stride, stride, 1], padding='SAME')
        output_shape = output.get_shape()
        output = tf.reshape(output, shape=[int(output_shape[0]), int(
            output_shape[1]), int(output_shape[2]), int(input_shape[3]), kernel * kernel])
        output = tf.transpose(output, perm=[0, 1, 2, 4, 3])
    
        return output   
    
    def dilation(self, input, input_dim, output_dim, name):
        with tf.variable_scope(name) as scope:
            a = tf.nn.relu(self.Atrous_conv2d(input, [3, 3, input_dim, output_dim], 1, 0.01, name = 'dilation1'))
            b = tf.nn.relu(self.Atrous_conv2d(input, [3, 3, input_dim, output_dim], 3, 0.01, name = 'dilation3'))
            c = tf.nn.relu(self.Atrous_conv2d(input, [3, 3, input_dim, output_dim], 5, 0.01, name = 'dilation5'))
            d = tf.nn.relu(self.Atrous_conv2d(input, [3, 3, input_dim, output_dim], 7, 0.01, name = 'dilation7'))
            e = tf.concat([a, b, c, d], axis = 3)
            
        return e
    
    def Atrous_conv2d(self, input, shape, rate, stddev, name, padding = 'SAME'):
        with tf.variable_scope(name):
            W = tf.get_variable('W',
                                shape = shape,
                                initializer = tf.truncated_normal_initializer(stddev = stddev))
            atrous_conv = tf.nn.atrous_conv2d(input, W, rate = rate, padding = padding)
            b = tf.get_variable('b', shape = [shape[3]], initializer = tf.constant_initializer(0.0))
            atrous_conv = tf.nn.bias_add(atrous_conv, b)
            
        return atrous_conv    
    
    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)    
    
    
    def mat_transform(self, input, caps_num_c, regularizer, s, tag=False):
        batch_size = int(input.get_shape()[0])
        caps_num_i = int(input.get_shape()[1])
        output = tf.reshape(input, shape=[batch_size, caps_num_i, 1, 4, 4])
        # the output of capsule is miu, the mean of a Gaussian, and activation, the sum of probabilities
        # it has no relationship with the absolute values of w and votes
        # using weights with bigger stddev helps numerical stability
        w = slim.variable('w' + str(s), shape=[1, caps_num_i, caps_num_c, 4, 4], dtype=tf.float32,
                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0),
                          regularizer=regularizer)
    
        w = tf.tile(w, [batch_size, 1, 1, 1, 1])
        output = tf.tile(output, [1, 1, caps_num_c, 1, 1])
        votes = tf.reshape(tf.matmul(output, w), [batch_size, caps_num_i, caps_num_c, 16])
    
        return votes
    
    def em_routing(self, votes, activation, caps_num_c, regularizer, s, tag=False):
        test = []
    
        batch_size = int(votes.get_shape()[0])
        caps_num_i = int(activation.get_shape()[1])
        n_channels = int(votes.get_shape()[-1])
    
        sigma_square = []
        miu = []
        activation_out = []
        beta_v = slim.variable('beta_v' + str(s), shape=[caps_num_c, n_channels], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0), regularizer=regularizer)
        beta_a = slim.variable('beta_a' + str(s), shape=[caps_num_c], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0), regularizer=regularizer)
    
        # votes_in = tf.stop_gradient(votes, name='stop_gradient_votes')
        # activation_in = tf.stop_gradient(activation, name='stop_gradient_activation')
        votes_in = votes
        activation_in = activation
    
        for iters in range(cfg.iter_routing):
            # if iters == cfg.iter_routing-1:
    
            # e-step
            if iters == 0:
                r = tf.constant(np.ones([batch_size, caps_num_i, caps_num_c], dtype=np.float32) / caps_num_c)
            else:
                # Contributor: Yunzhi Shi
                # log and exp here provide higher numerical stability especially for bigger number of iterations
                log_p_c_h = -tf.log(tf.sqrt(sigma_square)) - \
                            (tf.square(votes_in - miu) / (2 * sigma_square))
                log_p_c_h = log_p_c_h - \
                            (tf.reduce_max(log_p_c_h, axis=[2, 3], keep_dims=True) - tf.log(10.0))
                p_c = tf.exp(tf.reduce_sum(log_p_c_h, axis=3))
    
                ap = p_c * tf.reshape(activation_out, shape=[batch_size, 1, caps_num_c])
    
                # ap = tf.reshape(activation_out, shape=[batch_size, 1, caps_num_c])
    
                r = ap / (tf.reduce_sum(ap, axis=2, keep_dims=True) + cfg.epsilon)
    
            # m-step
            r = r * activation_in
            r = r / (tf.reduce_sum(r, axis=2, keep_dims=True)+cfg.epsilon)
    
            r_sum = tf.reduce_sum(r, axis=1, keep_dims=True)
            r1 = tf.reshape(r / (r_sum + cfg.epsilon),
                            shape=[batch_size, caps_num_i, caps_num_c, 1])
    
            miu = tf.reduce_sum(votes_in * r1, axis=1, keep_dims=True)
            sigma_square = tf.reduce_sum(tf.square(votes_in - miu) * r1,
                                         axis=1, keep_dims=True) + cfg.epsilon
    
            if iters == cfg.iter_routing-1:
                r_sum = tf.reshape(r_sum, [batch_size, caps_num_c, 1])
                cost_h = (beta_v + tf.log(tf.sqrt(tf.reshape(sigma_square,
                                                             shape=[batch_size, caps_num_c, n_channels])))) * r_sum
    
                activation_out = tf.nn.softmax(cfg.ac_lambda0 * (beta_a - tf.reduce_sum(cost_h, axis=2)))
            else:
                activation_out = tf.nn.softmax(r_sum)
            # if iters <= cfg.iter_routing-1:
            #     activation_out = tf.stop_gradient(activation_out, name='stop_gradient_activation')
    
        return miu, activation_out, test
    
    def primary(self, input1, batch, dim, data_size, s):
        with tf.variable_scope('primary_caps' + str(s)) as scope:
            pose = slim.conv2d(input1, num_outputs=dim * 16,
                               kernel_size=[1, 1], stride=1, padding='VALID', scope='primary_caps/pose' + str(s), activation_fn=None)
            activation = slim.conv2d(input1, num_outputs=dim, kernel_size=[
                                     1, 1], stride=1, padding='VALID', scope='primary_caps/activation' + str(s), activation_fn=tf.nn.sigmoid)
            pose = tf.reshape(pose, shape=[batch, data_size, data_size, dim, 16])
            activation = tf.reshape(
                activation, shape=[batch, data_size, data_size, dim, 1])
            output = tf.concat([pose, activation], axis=4)     
            
        return output
    
    def convcaps1(self, input1, batch, data_size, dim1, dim2, dim_trans, weights_regularizer, s):
        with tf.variable_scope('conv_caps5') as scope:
            output = tf.reshape(input1, shape=[batch, data_size, data_size, -1])
            output = self.kernel_tile(output, 3, 1)
            output_temp = tf.reshape(output, shape=[batch, data_size, data_size, -1]) 
            output = tf.reshape(output, shape=[batch *
                                                                     data_size * data_size, 3 * 3 * dim1, 17])
            activation_column = self.activation_trans_column(output[:, :, 16], data_size, dim1, dim_trans, weights_regularizer, s)
            activation_row = self.activation_trans_row(output[:, :, 16], data_size, dim1, dim_trans, weights_regularizer, s)                 
        
            with tf.variable_scope('v') as scope:
                with tf.variable_scope('column') as scope:
                    pose_column = self.pose_trans_column(output[:, :, :16], data_size, dim1, dim_trans, weights_regularizer, s)
                    votes_column = self.mat_transform(pose_column, dim2, weights_regularizer, s, tag=True)
        
                with tf.variable_scope('row') as scope:
                    pose_row = self.pose_trans_row(output[:, :, :16], data_size, dim1, dim_trans, weights_regularizer, s)
                    votes_row = self.mat_transform(pose_row, dim2, weights_regularizer, s, tag=True)                    
        
            with tf.variable_scope('routing') as scope:
                with tf.variable_scope('column') as scope:
                    miu_column, activation_column, _ = self.em_routing(votes_column, activation_column, dim2, weights_regularizer, s)
        
                with tf.variable_scope('row') as scope:
                    miu_row, activation_row, _ = self.em_routing(votes_row, activation_row, dim2, weights_regularizer, s) 

                miu = self.miu_routing_trans(miu_column, miu_row, 1, data_size, dim_trans, dim2, 1)
        
                activation = self.activation_routing_trans(activation_column, activation_row, 1, data_size, dim_trans, dim2, 1)                 
        
            pose = tf.reshape(miu, shape=[batch, data_size, data_size, dim2, 16])
            activation = tf.reshape(
                            activation, shape=[batch, data_size, data_size, dim2, 1])   
            output = tf.reshape(tf.concat([pose, activation], axis=4), [batch, data_size, data_size, -1]) 
            #self.output = self.Conv_2d(tf.concat([self.output, self.output_temp], axis=3), [1, 1, (cfg.C * 17 + 3 * 3 * cfg.B * 17), cfg.C * 17], 0.01, padding='VALID', name='conv1_Concat')  
            
        return output
    
    def convcaps2(self, input1, batch, data_size, dim1, dim2, dim_trans, weights_regularizer, s):
        with tf.variable_scope('conv_caps5') as scope:
            output = tf.reshape(input1, shape=[batch, data_size, data_size, -1])
            output = self.kernel_tile(output, 3, 2)
            data_size = int(np.floor(data_size / 2))
            output_temp = tf.reshape(output, shape=[batch, data_size, data_size, -1])  
            output = tf.reshape(output, shape=[batch *
                                                                     data_size * data_size, 3 * 3 * dim1, 17])
            activation_column = self.activation_trans_column(output[:, :, 16], data_size, dim1, dim_trans, weights_regularizer, s)
            activation_row = self.activation_trans_row(output[:, :, 16], data_size, dim1, dim_trans, weights_regularizer, s)                 
        
            with tf.variable_scope('v') as scope:
                with tf.variable_scope('column') as scope:
                    pose_column = self.pose_trans_column(output[:, :, :16], data_size, dim1, dim_trans, weights_regularizer, s)
                    votes_column = self.mat_transform(pose_column, dim2, weights_regularizer, s, tag=True)
        
                with tf.variable_scope('row') as scope:
                    pose_row = self.pose_trans_row(output[:, :, :16], data_size, dim1, dim_trans, weights_regularizer, s)
                    votes_row = self.mat_transform(pose_row, dim2, weights_regularizer, s, tag=True)                    
        
            with tf.variable_scope('routing') as scope:
                with tf.variable_scope('column') as scope:
                    miu_column, activation_column, _ = self.em_routing(votes_column, activation_column, dim2, weights_regularizer, s)
        
                with tf.variable_scope('row') as scope:
                    miu_row, activation_row, _ = self.em_routing(votes_row, activation_row, dim2, weights_regularizer, s) 

                miu = self.miu_routing_trans(miu_column, miu_row, 1, data_size, dim_trans, dim2, 1)
        
                activation = self.activation_routing_trans(activation_column, activation_row, 1, data_size, dim_trans, dim2, 1)                 
        
            pose = tf.reshape(miu, shape=[batch, data_size, data_size, dim2, 16])
            activation = tf.reshape(
                            activation, shape=[batch, data_size, data_size, dim2, 1])   
            output = tf.reshape(tf.concat([pose, activation], axis=4), [batch, data_size, data_size, -1]) 
            #self.output = self.Conv_2d(tf.concat([self.output, self.output_temp], axis=3), [1, 1, (cfg.C * 17 + 3 * 3 * cfg.B * 17), cfg.C * 17], 0.01, padding='VALID', name='conv1_Concat')  
            
        return output    
    


if __name__ == "__main__":

    img = cv2.imread("E:/LY/NLDF-master/dataset/img/1.jpg")

    h, w = img.shape[0:2]
    img = cv2.resize(img, (img_size,img_size)) - vgg16.VGG_MEAN
    img = img.reshape((1, img_size, img_size, 3))

    label = cv2.imread("E:/LY/NLDF-master/dataset/label/1.png")[:, :, 0]
    label = cv2.resize(label, (label_size, label_size))
    label = label.astype(np.float32) / 255
    label = np.stack((label, 1-label), axis=2)
    label = np.reshape(label, [-1, 2])

    sess = tf.Session()

    model = Model()
    model.build_model()

    max_grad_norm = 1
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(model.C_IoU_LOSS, tvars), max_grad_norm)
    opt = tf.train.AdamOptimizer(1e-5)
    optimizer = opt.apply_gradients(zip(grads, tvars))

    sess.run(tf.global_variables_initializer())

    for i in range(200):  #python2.x xrange, python3.x range
        _, C_IoU_LOSS = sess.run([optimizer, model.C_IoU_LOSS],
                                 feed_dict={model.input_holder: img,
                                            model.label_holder: label})

        print('[Iter %d] Contour Loss: %f' % (i, C_IoU_LOSS))

    boundary, gt_boundary = sess.run([model.Prob_Grad, model.label_Grad],
                                     feed_dict={model.input_holder: img,
                                                model.label_holder: label})

    boundary = np.squeeze(boundary)
    boundary = cv2.resize(boundary, (w, h))

    gt_boundary = np.squeeze(gt_boundary)
    gt_boundary = cv2.resize(gt_boundary, (w, h))

    cv2.imshow('boundary', np.uint8(boundary*255))
    cv2.imshow('boundary_gt', np.uint8(gt_boundary*255))

    cv2.waitKey()
