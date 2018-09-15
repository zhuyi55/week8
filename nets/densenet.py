"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1_' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net

def densenet(images, num_classes=1001, is_training=False, dropout_keep_prob=0.8, prediction_fn=slim.softmax, scope='densenet'):
    print ("==========hello densenet==========")
    growth = 24
    #growth = 12
    compression_rate = 0.5
    # compression_rate = 0.8 

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            #pass
            ##########################
            # Put your code here.
            ##########################

            #convolution with 16 output channels.
            # 32 * 32 * 3
            end_point = "densnet_1"
            net = slim.conv2d(images, growth*2, [3, 3], padding='SAME', stride=1, scope=end_point)
            end_points[end_point] = net
            print('{0}:{1}'.format(end_point, net.shape))

            #32*32*48
            with tf.variable_scope('dense_block_1'):
                end_point = "densnet_2"
                net = block(net, 40, growth)
                end_points[end_point] = net
            print('{0}:{1}'.format(end_point, net.shape))

            
            #32*32*12
            with tf.variable_scope('transition_layer_1'):
                end_point = "densnet_3"
                net = slim.batch_norm(net, scope=scope + '_bn')
                net = slim.conv2d(net, 2*growth, [1, 1], padding='SAME', scope=scope + '_conv')
                net = slim.avg_pool2d(net, [2, 2], stride=2, padding='VALID', scope= scope +'AvgPool')
                end_points[end_point] = net
            print('{0}:{1}'.format(end_point, net.shape))

            #16*16
            with tf.variable_scope('dense_block_2'):
                end_point = "densnet_4"
                net = block(net, 40, growth)
                end_points[end_point] = net
            print('{0}:{1}'.format(end_point, net.shape))
            
            #16*16*24
            with tf.variable_scope('transition_layer_2'):
                end_point = "densnet_5"
                net = slim.batch_norm(net, scope=scope + '_bn')
                net = slim.conv2d(net, 2*growth, [1, 1], padding='SAME', scope=scope + '_conv')
                net = slim.avg_pool2d(net, [2, 2], stride=2, padding='VALID', scope= scope +'AvgPool')
                end_points[end_point] = net
            print('{0}:{1}'.format(end_point, net.shape))

            #8*8*24
            with tf.variable_scope('dense_block_3'):
                end_point = "densnet_6"
                net = block(net, 40, growth)
                end_points[end_point] = net
            print('{0}:{1}'.format(end_point, net.shape))
            
            #8*8*40
            with tf.variable_scope('global'):
                end_point = "densnet_7"
                # net = slim.avg_pool2d(net, [8, 8], stride=1, padding='VALID', scope= scope +'_AvgPool')
                # end_points[end_point] = net
                
                # Pooling with a fixed kernel size.
                logits = slim.avg_pool2d(net, net.shape[1:3], padding='VALID', scope= scope +'_AvgPool')
                print('{0}:{1}'.format(end_point, logits.shape))

            with tf.name_scope('global_end'):
                end_point = "densnet_8"
                logits = tf.squeeze(tf.contrib.slim.conv2d(logits, num_classes, [1,1], activation_fn=None))
                end_points[end_point] = logits
            print('{0}:{1}'.format(end_point, logits.shape))

            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
            end_points[end_point] = logits

    return logits, end_points

def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 32
