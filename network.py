import numpy as np
import tensorflow as tf

DEFAULT_PADDING = 'SAME'

def load(data_path, session):
    data_dict = np.load(data_path).item()
    for key in data_dict:
        with tf.variable_scope(key, reuse=True):
            for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                session.run(tf.get_variable(subkey).assign(data))

def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path).item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))

def get_unique_name(self,prefix):
    id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
    return '%s_%d'%(prefix, id)


def save_npy(self, sess, npy_path="./vgg16_retrain.npy"):
    assert isinstance(sess, tf.Session)

    data_dict = {}

    for (name, idx), var in self.var_dict.items():
        var_out = sess.run(var)
        if not data_dict.has_key(name):
            data_dict[name] = {}
        data_dict[name][idx] = var_out

    np.save(npy_path, data_dict)
    print("file saved", npy_path)
    return npy_path

def make_var(name, shape):
    return tf.get_variable(name, shape)

def conv(input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1):
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0        
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        kernel = make_var('weights', shape=[k_h, k_w, c_i/group, c_o])
        biases = make_var('biases', [c_o])
        if group==1:
            conv = convolve(input, kernel)
        else:
            input_groups = tf.split(3, group, input)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
            conv = tf.concat(3, output_groups)                
        if relu:
            bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
            return tf.nn.relu(bias, name=scope.name)
        return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list(), name=scope.name)

def relu(input, name):
    return tf.nn.relu(input, name=name)

def max_pool(input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
    return tf.nn.max_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)

def avg_pool(input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
    return tf.nn.avg_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)

def lrn(input, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(input,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias,
                                              name=name)

def concat(inputs, axis, name):
    return tf.concat(concat_dim=axis, values=inputs, name=name)

def fc(input, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:
        weights = make_var('weights', shape=[num_in, num_out])
        biases = make_var('biases', [num_out])
        op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
        fc = op(input, weights, biases, name=scope.name)
        return fc

def softmax(input, name):
    return tf.nn.softmax(input, name)

def dropout(input, keep_prob):
    return tf.nn.dropout(input, keep_prob)

def deconv2d(inputs, filter_height, filter_width, output_shape, stride=(1, 1), padding='SAME', name='Deconv2D'):
    input_channels = int(inputs.get_shape()[-1])
    output_channels = output_shape[-1]
    # fan_in = filter_height * filter_width * output_channels
    # stddev = tf.sqrt(2.0 / fan_in)
    weights_shape = [filter_height, filter_width, output_channels, input_channels]
    biases_shape = [output_channels]

    with tf.variable_scope(name):
        # filters_init = tf.truncated_normal_initializer(stddev=stddev)
        # biases_init = tf.constant_initializer(0.1)

        filters = tf.get_variable(
            'weights', shape=weights_shape, collections=['weights', 'variables'])
            # 'weights', shape = weights_shape, initializer = filters_init, collections = ['weights', 'variables'])
        # biases = tf.get_variable(
            # 'biases', shape=biases_shape, collections=['biases', 'variables'])
            # 'biases', shape = biases_shape, initializer = biases_init, collections = ['biases', 'variables'])
        deconv=tf.nn.conv2d_transpose(inputs, filters, output_shape, strides=[1, 2, 2, 1], padding=padding)
        deconv = tf.nn.relu(deconv)
        return deconv
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

def maxpool3d(inputs, depth_k=2,k=2,name='maxpool3d'):
    # MaxPool2D wrapper
    return tf.nn.max_pool3d(inputs, ksize=[1, depth_k, k, k, 1], strides=[1, depth_k, k, k, 1],
                          padding='SAME',name=name)


def conv3d(inputs, filters, bias, padding='SAME',strides=1,name='conv3d'):
    # with tf.variable_scope(name):
    # filters = tf.get_variable(
    #     'weights', shape=filters, collections=['weights', 'variables'])
    # bias = tf.get_variable(
    #     'weights', shape=bias, collections=['weights', 'variables'])
    # filters=make_var('weights',shape=filters)
    # bias=make_var('bias',shape=bias)
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            filters_shape =  tf.Variable(tf.truncated_normal(shape=filters,stddev=0.1))
        with tf.name_scope('biases'):
            bias_shape = tf.Variable(tf.truncated_normal(shape=bias,stddev=0.1))
        with tf.name_scope('conv3d'):
            conv3d = tf.nn.conv3d(inputs, filter=filters_shape, strides=[1, strides, strides, strides, 1], padding=padding,name=name)
        with tf.name_scope('bias_add'):
            outputs = tf.nn.bias_add(conv3d, bias=bias_shape)
        conv3d = tf.nn.relu(outputs)
        tf.summary.histogram(name + '/outputs', conv3d)
    return conv3d

def deconv3d(inputs, filters, bias, output_shape, strides,padding='VALID',name='deconv3d'):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            filters_shape = tf.Variable(tf.truncated_normal(shape=filters,stddev=0.1))
        with tf.name_scope('biases'):
            bias_shape = tf.Variable(tf.truncated_normal(shape=bias,stddev=0.1))
    # filters = tf.get_variable(
    #     'weights', shape=filters, collections=['weights', 'variables'])
    # bias = tf.get_variable(
    #     'weights', shape=bias, collections=['weights', 'variables'])
    # filters=make_var('weights',shape=filters)
    # bias=make_var('bias',shape=bias)
        with tf.name_scope('deconv'):
            deconv = tf.nn.conv3d_transpose(inputs, filter=filters_shape, output_shape=output_shape, strides=strides, padding=padding,name=name)
        with tf.name_scope('bias_add'):
            deconv = tf.nn.bias_add(deconv, bias=bias_shape)
        deconv = tf.nn.relu(deconv)
        tf.summary.histogram(name + '/outputs', deconv)
    return deconv

def leaky_relu(inputs, leak=0.1, name='LeakyRelu'):
    with tf.name_scope(name):
        return tf.maximum(inputs, leak * inputs)

def batch_norm(inputs, decay, is_training, var_epsilon=1e-3, name='batch_norm'):
    with tf.variable_scope(name):
        scale = tf.Variable(tf.ones([int(inputs.get_shape()[-1])]))
        offset = tf.Variable(tf.zeros([int(inputs.get_shape()[-1])]))
        avg_mean = tf.Variable(tf.zeros([int(inputs.get_shape()[-1])]), trainable=False)
        avg_var = tf.Variable(tf.ones([int(inputs.get_shape()[-1])]), trainable=False)

        def get_batch_moments():
            batch_mean, batch_var = tf.nn.moments(inputs, list(range(len(inputs.get_shape()) - 1)))
            assign_mean = tf.assign(avg_mean, decay * avg_mean + (1.0 - decay) * batch_mean)
            assign_var = tf.assign(avg_var, decay * avg_var + (1.0 - decay) * batch_var)
            with tf.control_dependencies([assign_mean, assign_var]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        def get_avg_moments():
            return avg_mean, avg_var

        mean, var = tf.cond(is_training, get_batch_moments, get_avg_moments)
        return tf.nn.batch_normalization(inputs, mean, var, offset, scale, var_epsilon)


# def batch_norm(x, is_training=True, scope='batch_norm'):
#     return tf_contrib.layers.batch_norm(x,
#                                         decay=0.9, epsilon=1e-05,
#                                         center=True, scale=True, updates_collections=None,
#                                         is_training=is_training, scope=scope)


def attention(x, ch, name, mask=False):
    ch = x.get_shape()[-1]    
    with tf.variable_scope(name): # [bs, h, w, c]
        f = tf.layers.conv3d(x, filters=ch//8, # [bs, l, h, w, c']
            kernel_size=1,
            strides=1,
            padding='SAME',
        )

        g = tf.layers.conv3d(x, filters=ch//8, # [bs, l, h, w, c']
            kernel_size=1,
            strides=1,
            padding='SAME',
        )


        h = tf.layers.conv3d(x, filters=ch, # [bs, l, h, w, c']
            kernel_size=1,
            strides=1,
            padding='SAME',
        )
    
    # print f, g, 
    # N = l * h * w 
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]
    beta = tf.nn.softmax(s, axis=-1)  # attention map

    o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
    gamma = tf.get_variable("gamma"+name, [1], initializer=tf.constant_initializer(0.0))

    o = tf.reshape(o, shape=tf.shape(x)) # [bs, h, w, C]
    if mask:
        x = o * gamma
    else:
        x = o * gamma +x
    return x

# opsnput.get_shape().as_list()[1]
def hw_flatten(x) :
    return tf.reshape(x, shape=[tf.shape(x)[0], -1, tf.shape(x)[-1]])

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
    