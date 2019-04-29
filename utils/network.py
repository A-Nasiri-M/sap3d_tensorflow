import numpy as np
import tensorflow as tf

DEFAULT_PADDING = 'SAME'
# POOL
def pool3d(value, sub_size):
    return tf.layers.max_pooling3d(value, sub_size, sub_size)

def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], 1)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out

def unpool3D(value,input_shape, stides=(1,2,2), name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:

        # ch = input_shape.get_shape().as_list()
        value = (tf.reshape(value, [input_shape[0]]+[input_shape[2]]+[input_shape[3]]+[input_shape[4]]))
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])  #from 2 dimentaion
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        # print dim, sh
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        # out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out_size = [-1] + [1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out

# loss
def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=3.0, dim=[0]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff

    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))

    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box))
    # loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box))
    return loss_box

# BN / GN
def GroupNorm(x, G=32, esp=1e-5):
    """
    https://arxiv.org/abs/1803.08494
    """
    with tf.variable_scope('group_norm'):
        # normalize
        # tranpose: [bs, d, h, w, c] to [bs, c, d, h, w] following the paper
        x = tf.transpose(x, [0, 4, 1, 2, 3])
        N, C, D, H, W = x.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, D, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4, 5], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + esp)
        # per channel gamma and beta
        gamma = tf.Variable(tf.constant(1.0, shape=[C]), dtype=tf.float32, name='gamma')
        beta = tf.Variable(tf.constant(0.0, shape=[C]), dtype=tf.float32, name='beta')
        gamma = tf.reshape(gamma, [1, C, 1, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1, 1])

        output = tf.reshape(x, [-1, C, D, H, W]) * gamma + beta
        # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
        output = tf.transpose(output, [0, 2, 3, 4, 1])
    return output

def normalize(x, training, mode='bn'):
    if mode == 'bn':
        x = tf.layers.batch_normalization(x, training=training)
    elif mode == 'gn':
        x = GroupNorm(x)
    return x

# layers
def concat(x):
    return tf.concat(x, axis=-1)

def conv3d(x, channel, kernel, strides, training, name, mode='bn'):
    x = tf.layers.conv3d(x, channel, kernel, strides, 'same', name=name)
    x = normalize(x, training, mode)
    x = tf.nn.relu(x)
    return x

def transpose_conv3d(x, channel, kernel, strides, training, name, mode='bn'):
    x = tf.layers.conv3d_transpose(x, channel, kernel, strides, 'same', name=name)
    x = normalize(x, training, mode)
    x = tf.nn.relu(x)
    return x

## Attention
## NON-LOCAL
def non_local(x, name, training, sub_sample=True):
    residual = x
    in_channels = x.get_shape()[-1]
    inter_channels =  in_channels // 2
    batch_size = x.get_shape().as_list()[0]
    g_x = tf.layers.conv3d(x, filters=inter_channels, # [bs, l, h, w, c']
            kernel_size=1,
            strides=1,
            padding='SAME',
        )
    if sub_sample:
        g_x = pool3d(g_x)
    g_x = tf.reshape(g_x, [batch_size, -1, inter_channels])

    theta_x = tf.layers.conv3d(x, filters=inter_channels, # [bs, l, h, w, c']
            kernel_size=1,
            strides=1,
            padding='SAME',
        )
    theta_x = tf.reshape(theta_x, (batch_size, -1, inter_channels))

    phi_x = tf.layers.conv3d(x, filters=inter_channels, # [bs, l, h, w, c']
            kernel_size=1,
            strides=1,
            padding='SAME',
        )
    if sub_sample:
        phi_x = pool3d(phi_x)
    phi_x = tf.reshape(phi_x, (batch_size, inter_channels, -1))

    f = tf.matmul(theta_x, phi_x)
    N = f.get_shape().as_list()[-1]
    f_div_C = f / N
    y = tf.matmul(f_div_C, g_x)
    y = tf.reshape(y, [batch_size] + x.get_shape().as_list()[1:-1] + [inter_channels])
    W_y = tf.layers.conv3d(y, in_channels, 1, 1, 'same')
    W_y = normalize(W_y, training=training, mode='bn')
    W_y = tf.nn.relu(W_y)
    z = W_y + residual

    return z

# SA
def attention(x, name, training, mode='bn', subsample=False, sub_size=2):
    # ch = x.get_shape()[-1]
    shape = x.get_shape().as_list()
    batch_size = shape[0]
    ch = shape[-1]
    inter_channels = max(1, ch//8)
    with tf.variable_scope(name): # [bs, h, w, c]
        f = tf.layers.conv3d(x, filters=inter_channels, # [bs, l, h, w, c']
            kernel_size=1,
            strides=1,
            padding='SAME',
        )
        g = tf.layers.conv3d(x, filters=inter_channels, # [bs, l, h, w, c']
            kernel_size=1,
            strides=1,
            padding='SAME',
        )
        h = tf.layers.conv3d(x, filters=ch, # [bs, l, h, w, c]
            kernel_size=1,
            strides=1,
            padding='SAME',
        )
    # N = l * h * w 
    if subsample:
        f = pool3d(f, sub_size)
        g = pool3d(g, sub_size/2)
        h = pool3d(h, sub_size)
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N/8]
    beta = tf.nn.softmax(s, axis=-1)  # attention map
    o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
    o = tf.reshape(o, shape=[batch_size]+[each*2/sub_size for each in shape[1:-1]]+[ch]) # [bs, h, w, C]
    o = tf.layers.conv3d(o, ch, 1, sub_size/2, 'same')
    o = normalize(o, training=training, mode=mode)
    o = tf.nn.relu(o)
    gamma = tf.get_variable("gamma"+name, [1], initializer=tf.constant_initializer(0.0))
    x = o * gamma +x
    return x

def hw_flatten(x) :
    return tf.reshape(x, shape=[tf.shape(x)[0], -1, tf.shape(x)[-1]])
# CBAM
def cbam_block(input_feature, name, ratio=8):
  """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
  As described in https://arxiv.org/abs/1807.06521.
  """

  with tf.variable_scope(name):
    attention_feature = channel_attention(input_feature, 'ch_at', ratio)
    attention_feature = spatial_attention(attention_feature, 'sp_at')
  return attention_feature

def channel_attention(input_feature, name, ratio=8):
  
  kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
  bias_initializer = tf.constant_initializer(value=0.0)
  
  with tf.variable_scope(name):
    
    channel = input_feature.get_shape()[-1]
    avg_pool = tf.reduce_mean(input_feature, axis=[1,2,3], keepdims=True)
    assert avg_pool.get_shape()[1:] == (1,1,1,channel)
    avg_pool = tf.layers.dense(inputs=avg_pool,
                                 units=channel//ratio,
                                 activation=tf.nn.relu,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='mlp_0',
                                 reuse=None)   
    assert avg_pool.get_shape()[1:] == (1,1,1,channel//ratio)
    avg_pool = tf.layers.dense(inputs=avg_pool,
                                 units=channel,                             
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='mlp_1',
                                 reuse=None)    
    assert avg_pool.get_shape()[1:] == (1,1,1,channel)

    max_pool = tf.reduce_max(input_feature, axis=[1,2,3], keepdims=True)
    assert max_pool.get_shape()[1:] == (1,1,1,channel)
    max_pool = tf.layers.dense(inputs=max_pool,
                                 units=channel//ratio,
                                 activation=tf.nn.relu,
                                 name='mlp_0',
                                 reuse=True)   
    assert max_pool.get_shape()[1:] == (1,1,1,channel//ratio)
    max_pool = tf.layers.dense(inputs=max_pool,
                                 units=channel,                             
                                 name='mlp_1',
                                 reuse=True)  
    assert max_pool.get_shape()[1:] == (1,1,1,channel)
    scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')
    
  return input_feature * scale

def spatial_attention(input_feature, name):
  kernel_size = 7
  kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
  with tf.variable_scope(name):
    avg_pool = tf.reduce_mean(input_feature, axis=[4], keepdims=True)
    assert avg_pool.get_shape()[-1] == 1
    max_pool = tf.reduce_max(input_feature, axis=[4], keepdims=True)
    assert max_pool.get_shape()[-1] == 1
    concat = tf.concat([avg_pool,max_pool], 4)
    assert concat.get_shape()[-1] == 2
    
    concat = tf.layers.conv3d(concat,
                              filters=1,
                              kernel_size=[kernel_size,kernel_size,kernel_size],
                              strides=[1,1,1],
                              padding="same",
                              activation=None,
                              kernel_initializer=kernel_initializer,
                              use_bias=False,
                              name='conv3d')
    assert concat.get_shape()[-1] == 1
    concat = tf.sigmoid(concat, 'sigmoid')
    
  return input_feature * concat
    
# CBAM-Temporal
