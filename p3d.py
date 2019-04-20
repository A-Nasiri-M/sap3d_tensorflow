import tensorflow as tf
from utils.network import *
# from settings import *
CROP_SIZE=112  
NUM_FRAMES_PER_CLIP=16  #clip length
BATCH_SIZE=12
RGB_CHANNEL=3
BLOCK_EXPANSION=4  #You do not need change

def get_conv_weight(name,kshape,wd=0.0005):
    with tf.device('/cpu:0'):
        var=tf.get_variable(name,shape=kshape,initializer=tf.contrib.layers.xavier_initializer())
    if wd!=0:
        weight_decay = tf.nn.l2_loss(var)*wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var

def convS(name,l_input,in_channels,out_channels):
    return tf.nn.bias_add(tf.nn.conv3d(l_input,get_conv_weight(name=name,
                                                               kshape=[1,3,3,in_channels,out_channels]),
                                                               strides=[1,1,1,1,1],padding='SAME'),
                                              get_conv_weight(name+'_bias',[out_channels],0))
def convT(name,l_input,in_channels,out_channels):
    return tf.nn.bias_add(tf.nn.conv3d(l_input,get_conv_weight(name=name,
                                                               kshape=[3,1,1,in_channels,out_channels]),
                                                               strides=[1,1,1,1,1],padding='SAME'),
                                              get_conv_weight(name+'_bias',[out_channels],0))

#build the bottleneck struction of each block.
class Bottleneck():
    def __init__(self,l_input,inplanes,planes,stride=1,downsample='', training=True, n_s=0,depth_3d=47):
        
        self.X_input=l_input
        self.downsample=downsample
        self.planes=planes
        self.inplanes=inplanes
        self.depth_3d=depth_3d
        self.ST_struc=('A','B','C')
        self.len_ST=len(self.ST_struc)
        self.id=n_s
        self.n_s=n_s
        self.ST=list(self.ST_struc)[self.id % self.len_ST]
        self.stride_p=[1,1,1,1,1]
        self.training = training
        if self.downsample!='':
            self.stride_p=[1,1,2,2,1]
        if n_s<self.depth_3d:
            if n_s==0:
                self.stride_p=[1,1,1,1,1]
        else:
            if n_s==self.depth_3d:
                self.stride_p=[1,2,2,2,1]
            else:
                self.stride_p=[1,1,1,1,1]
    #P3D has three types of bottleneck sub-structions.
    def ST_A(self,name,x):
        x=convS(name+'_S',x,self.planes,self.planes)
        x=tf.layers.batch_normalization(x,training=self.training)
        x=tf.nn.relu(x)
        x=convT(name+'_T',x,self.planes,self.planes)
        x=tf.layers.batch_normalization(x,training=self.training)
        x=tf.nn.relu(x)
        return x
    
    def ST_B(self,name,x):
        tmp_x=convS(name+'_S',x,self.planes,self.planes)
        tmp_x=tf.layers.batch_normalization(tmp_x,training=self.training)
        tmp_x=tf.nn.relu(tmp_x)
        x=convT(name+'_T',x,self.planes,self.planes)
        x=tf.layers.batch_normalization(x,training=self.training)
        x=tf.nn.relu(x)
        return x+tmp_x
    
    def ST_C(self,name,x):
        x=convS(name+'_S',x,self.planes,self.planes)
        x=tf.layers.batch_normalization(x,training=self.training)
        x=tf.nn.relu(x)
        tmp_x=convT(name+'_T',x,self.planes,self.planes)
        tmp_x=tf.layers.batch_normalization(tmp_x,training=self.training)
        tmp_x=tf.nn.relu(tmp_x)
        return x+tmp_x
    
    def infer(self):
        residual=self.X_input
        if self.n_s<self.depth_3d:
            out=tf.nn.conv3d(self.X_input,get_conv_weight('conv3_{}_1'.format(self.id),[1,1,1,self.inplanes,self.planes]),
                             strides=self.stride_p,padding='SAME')
            out=tf.layers.batch_normalization(out,training=self.training)
            
        else:
            param=self.stride_p
            param.pop(1)
            out=tf.nn.conv2d(self.X_input,get_conv_weight('conv2_{}_1'.format(self.id),[1,1,self.inplanes,self.planes]),
                             strides=param,padding='SAME')
            out=tf.layers.batch_normalization(out,training=self.training)
    
        out=tf.nn.relu(out)    
        if self.id<self.depth_3d:
            if self.ST=='A':
                out=self.ST_A('STA_{}_2'.format(self.id),out)
            elif self.ST=='B':
                out=self.ST_B('STB_{}_2'.format(self.id),out)
            elif self.ST=='C':
                out=self.ST_C('STC_{}_2'.format(self.id),out)
        else:
            out=tf.nn.conv2d(out,get_conv_weight('conv2_{}_2'.format(self.id),[3,3,self.planes,self.planes]),
                                  strides=[1,1,1,1],padding='SAME')
            out=tf.layers.batch_normalization(out,training=self.training)
            out=tf.nn.relu(out)

        if self.n_s<self.depth_3d:
            out=tf.nn.conv3d(out,get_conv_weight('conv3_{}_3'.format(self.id),[1,1,1,self.planes,self.planes*BLOCK_EXPANSION]),
                             strides=[1,1,1,1,1],padding='SAME')
            out=tf.layers.batch_normalization(out,training=self.training)
        else:
            out=tf.nn.conv2d(out,get_conv_weight('conv2_{}_3'.format(self.id),[1,1,self.planes,self.planes*BLOCK_EXPANSION]),
                             strides=[1,1,1,1],padding='SAME')
            out=tf.layers.batch_normalization(out,training=self.training)
           
        if len(self.downsample)==1:
            residual=tf.nn.conv2d(residual,get_conv_weight('dw2d_{}'.format(self.id),[1,1,self.inplanes,self.planes*BLOCK_EXPANSION]),
                                  strides=[1,2,2,1],padding='SAME')
            residual=tf.layers.batch_normalization(residual,training=self.training)
        elif len(self.downsample)==2:
            residual=tf.nn.conv3d(residual,get_conv_weight('dw3d_{}'.format(self.id),[1,1,1,self.inplanes,self.planes*BLOCK_EXPANSION]),
                                  strides=self.downsample[1],padding='SAME')
            residual=tf.layers.batch_normalization(residual,training=self.training)
        
        # residual = attention(residual, ch=None, name='attention_{}'.format(self.id))
        # CBAM
        # residual = cbam_block(residual, name='attention_{}'.format(self.id))
        # attention for out
        out+=residual
        out=tf.nn.relu(out)
        
        return out

#build a singe block of p3d,depth_3d=47 means p3d-199
class make_block():
    def __init__(self,_X,planes,num,inplanes,cnt,training=True, depth_3d=47,stride=1):
        self.input=_X
        self.planes=planes
        self.inplanes=inplanes
        self.num=num
        self.cnt=cnt
        self.depth_3d=depth_3d
        self.stride=stride
        self.training=training
        if self.cnt<depth_3d:
            if self.cnt==0:
                stride_p=[1,1,1,1,1]
            else:
                stride_p=[1,1,2,2,1]
            if stride!=1 or inplanes!=planes*BLOCK_EXPANSION:
                self.downsample=['3d',stride_p]
        else:
            if stride!=1 or inplanes!=planes*BLOCK_EXPANSION:
                self.downsample=['2d']
    def infer(self):
        x=Bottleneck(self.input,self.inplanes,self.planes,self.stride,self.downsample,training=self.training, n_s=self.cnt,depth_3d=self.depth_3d).infer()
        self.cnt+=1
        self.inplanes=BLOCK_EXPANSION*self.planes
        for i in range(1,self.num):
            x=Bottleneck(x,self.inplanes,self.planes,training=self.training, n_s=self.cnt,depth_3d=self.depth_3d).infer()
            self.cnt+=1
        return x

#build structure of the p3d network.
def p3d_unet(_X,_dropout,batch_size=2, training=True):
    cnt=0
    # 16 112 112 3
    conv1_custom=tf.nn.conv3d(_X,get_conv_weight('firstconv1',[1,7,7,3,64]),strides=[1,1,2,2,1],padding='SAME')
    conv1_custom_bn=tf.layers.batch_normalization(conv1_custom,training=training)
    conv1_custom_bn_relu=tf.nn.relu(conv1_custom_bn)
    # 16 56 56 64
    pool1_concat=tf.nn.max_pool3d(conv1_custom_bn_relu,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    pool1=tf.nn.max_pool3d(conv1_custom_bn_relu,[1,2,3,3,1],strides=[1,2,2,2,1],padding='SAME')
    # 8 28 28 64
    b1=make_block(pool1,64,3,64,cnt)
    res1=b1.infer()
    # 8 28 28 256
    cnt=b1.cnt
    pool2=tf.nn.max_pool3d(res1,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    # 4 28 28 256
    b2=make_block(pool2,128,8,256,cnt,stride=2)
    res2=b2.infer()
    # 4 14 14 512 
    cnt=b2.cnt
    pool3=tf.nn.max_pool3d(res2,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    # 2 14 14 512
    b3=make_block(pool3,256,36,512,cnt,stride=2)
    res3=b3.infer()
    # 2 7 7 1024
    cnt=b3.cnt
    pool4=tf.nn.max_pool3d(res3,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    # 1 7 7 1024
    ###
    ### Deconvoltuion
    ###
    deconv1 = tf.layers.conv3d_transpose(pool4, 512, [1, 3, 3], [2, 2, 2], 'same')
    deconv1_bn = tf.layers.batch_normalization(deconv1, name='deconv1_bn', training=training)
    deconv1_re = tf.nn.relu(deconv1_bn)
    deconv1_concat = tf.concat([deconv1_re, pool3], axis=-1)
    # 2 14 14
    deconv2 = tf.layers.conv3d_transpose(deconv1_concat, 256, [2, 3, 3], [2, 2, 2], 'same')
    deconv2_bn = tf.layers.batch_normalization(deconv2, name='deconv2_bn', training=training)
    deconv2_re = tf.nn.relu(deconv2_bn)
    deconv2_concat = tf.concat([deconv2_re, pool2], axis=-1)
    # 4 28 28
    deconv3 = tf.layers.conv3d_transpose(deconv2_concat, 128, 3, [2, 2, 2], 'same')
    deconv3_bn = tf.layers.batch_normalization(deconv3, name='deconv3_bn', training=training)
    deconv3_re = tf.nn.relu(deconv3_bn)
    deconv3_concat = tf.concat([deconv3_re, pool1_concat], axis=-1)
    deconv3_drop = tf.layers.dropout(deconv3_re, rate=_dropout, training=training)
    # 8 56 56
    deconv4_conv1 = tf.layers.conv3d(deconv3_drop, 32, 1, 1, 'same')
    results = tf.layers.conv3d_transpose(deconv4_conv1, 1, 3, [2, 2, 2], 'same')
    # activation
    results = tf.sigmoid(results)
    # exit()
    return results


def p3d_concat(_X, _dropout, batch_size=2, training=True):
    cnt=0
    # 16 112 112 3
    conv1_custom=tf.nn.conv3d(_X,get_conv_weight('firstconv1',[1,7,7,3,64]),strides=[1,1,2,2,1],padding='SAME')
    conv1_custom_bn=tf.layers.batch_normalization(conv1_custom,training=training)
    conv1_custom_bn_relu=tf.nn.relu(conv1_custom_bn)
    # 16 56 56 64
    pool1=tf.nn.max_pool3d(conv1_custom_bn_relu,[1,2,3,3,1],strides=[1,2,2,2,1],padding='SAME')
    # 8 28 28 64
    b1=make_block(pool1,64,3,64,cnt)
    res1=b1.infer()
    # 8 28 28 256
    cnt=b1.cnt
    pool2=tf.nn.max_pool3d(res1,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    deconv_pool2 = tf.layers.conv3d_transpose(pool2, 128, 3, [1, 1, 1], 'same',name='deconv_pool2')
    deconv_pool2 = tf.layers.batch_normalization(deconv_pool2, name='deconv_pool2_bn', training=training)
    deconv_pool2 = tf.nn.relu(deconv_pool2)

    # 4 28 28 256
    b2=make_block(pool2,128,8,256,cnt,stride=2)
    res2=b2.infer()
    # 4 14 14 512 
    cnt=b2.cnt
    pool3=tf.nn.max_pool3d(res2,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    # 2 14 14 512
    # from 2 14 14 to 4 28 28 512
    deconv_pool3 = tf.layers.conv3d_transpose(pool3, 256, 3, [2, 2, 2], 'same',name='deconv_pool3')
    deconv_pool3 = tf.layers.batch_normalization(deconv_pool3, name='deconv_pool3_bn', training=training)
    deconv_pool3 = tf.nn.relu(deconv_pool3)

    b3=make_block(pool3,256,36,512,cnt,stride=2)
    res3=b3.infer()
    # 2 7 7 1024
    cnt=b3.cnt
    pool4=tf.nn.max_pool3d(res3,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    # 1 7 7 1024
    # from 1 7 7 to 4 28 28 1024
    deconv_pool4 = tf.layers.conv3d_transpose(pool4, 512, 3, [4, 4, 4], 'same',name='deconv_pool4')
    deconv_pool4 = tf.layers.batch_normalization(deconv_pool4, name='deconv_pool4_bn', training=training)
    deconv_pool4 = tf.nn.relu(deconv_pool4)
    ###
    ### Easy Upsampling
    ###
    concatenator = tf.concat([deconv_pool2, deconv_pool3, deconv_pool4], axis=-1, name='concatenator')
    conv_concat = tf.layers.conv3d(concatenator, 512, 3, 1, 'same', name='conv_concat')
    conv_concat = tf.layers.batch_normalization(conv_concat, name='conv_concat_bn', training=training)
    conv_concat = tf.nn.relu(conv_concat)
    deconv1_revise = tf.layers.conv3d_transpose(conv_concat, 128, 3, 2, 'same', name='deconv_revise')
    deconv1_revise = tf.layers.batch_normalization(deconv1_revise, name='deconv1_revise_bn', training=training)
    deconv1_revise = tf.nn.relu(deconv1_revise)
    deconv1_revise = tf.layers.dropout(deconv1_revise, rate=_dropout, training=training)
    results = tf.layers.conv3d_transpose(deconv1_revise, 1, 3, 2, 'same', name='predict_revise')
    return results


def concat(_X):
    return tf.concat(_X, axis=-1)

def conv3d(_X, channel, kernel, strides, training, name):
    _X = tf.layers.conv3d(_X, channel, kernel, strides, 'same', name=name)
    _X = tf.layers.batch_normalization(_X, training=training)
    _X = tf.nn.relu(_X)
    return _X

def transpose_conv3d(_X, channel, kernel, strides, training, name):
    _X = tf.layers.conv3d_transpose(_X, channel, kernel, strides, 'same', name=name)
    _X = tf.layers.batch_normalization(_X, training=training)
    _X = tf.nn.relu(_X)
    return _X

#build structure of the p3d network.
def p3d_unetplusplus(_X,_dropout,batch_size=2, training=True):
    cnt=0
    # 16 112 112 3
    conv1_custom=tf.nn.conv3d(_X,get_conv_weight('firstconv1',[1,7,7,3,64]),strides=[1,1,2,2,1],padding='SAME')
    conv1_custom_bn=tf.layers.batch_normalization(conv1_custom,training=training)
    conv1_custom_bn_relu=tf.nn.relu(conv1_custom_bn)
    # 16 56 56 64
    x_1_0=tf.nn.max_pool3d(conv1_custom_bn_relu,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    pool1=tf.nn.max_pool3d(conv1_custom_bn_relu,[1,2,3,3,1],strides=[1,2,2,2,1],padding='SAME')
    # 8 28 28 64
    b1=make_block(pool1,64,3,64,cnt)
    res1=b1.infer()
    # 8 28 28 256
    cnt=b1.cnt
    x_2_0=tf.nn.max_pool3d(res1,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    # 4 28 28 256
    b2=make_block(x_2_0,128,8,256,cnt,stride=2)
    res2=b2.infer()
    # 4 14 14 512 
    cnt=b2.cnt
    x_3_0=tf.nn.max_pool3d(res2,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    # 2 14 14 512
    b3=make_block(x_3_0,256,36,512,cnt,stride=2)
    res3=b3.infer()
    # 2 7 7 1024
    cnt=b3.cnt
    x_4_0=tf.nn.max_pool3d(res3,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    # 1 7 7 1024
    ###
    ### Deconvoltuion
    ###
    upx_4_0 = transpose_conv3d(x_4_0, 512, [1, 3, 3], [2, 2, 2], training, 'upx_4_0')
    x_3_1 = conv3d(concat([x_3_0, upx_4_0]), 512, [2, 3, 3], [1, 1, 1], training, 'x_3_1')

    upx_3_0 = transpose_conv3d(x_3_0, 256, [2, 3, 3], [2, 2, 2], training, 'upx_3_0')
    x_2_1 = conv3d(concat([x_2_0, upx_3_0]), 256, [3, 3, 3], [1, 1, 1], training, 'x_2_1')
    upx_3_1 = transpose_conv3d(x_3_1, 256, [2, 3, 3], [2, 2, 2], training, 'upx_3_1')
    x_2_2 = conv3d(concat([x_2_1, upx_3_1]), 256, [3, 3, 3], [1, 1, 1], training, 'x_2_2')

    upx_2_0 = transpose_conv3d(x_2_0, 128, [3, 3, 3], [2, 2, 2], training, 'upx_2_0')
    x_1_1 = conv3d(concat([x_1_0, upx_2_0]), 128, [3, 3, 3], [1, 1, 1], training, 'x_1_1')
    upx_2_1 = transpose_conv3d(x_2_1, 128, [3, 3, 3], [2, 2, 2], training, 'upx_2_1')
    x_1_2 = conv3d(concat([x_1_1, upx_2_1]), 128, [3, 3, 3], [1, 1, 1], training, 'x_1_2')
    upx_2_2 = transpose_conv3d(x_2_2, 128, [3, 3, 3], [2, 2, 2], training, 'upx_2_2')
    x_1_3 = conv3d(concat([x_1_2, upx_2_2]), 128, [3, 3, 3], [1, 1, 1], training, 'x_1_3')

    x_1_3 = tf.layers.dropout(x_1_3, _dropout, training=training)
    x_0_1 = tf.layers.conv3d_transpose(x_1_3, 1, 3, 2, 'same', name='x_0_1')

    return x_0_1
