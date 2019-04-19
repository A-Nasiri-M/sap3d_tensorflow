import tensorflow as tf
from network import *

# from settings import *
CROP_SIZE=112  
NUM_FRAMES_PER_CLIP=16  #clip length
BATCH_SIZE=12
RGB_CHANNEL=3
BLOCK_EXPANSION=4  #You do not need change

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


def GNReLU(x, name=None):
    x = GroupNorm(x, 32)
    return tf.nn.relu(x, name=name)


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

def deconvS(name, l_input, in_channels, out_channels):
    return tf.nn.bias_add(tf.nn.conv3d(l_input,get_conv_weight(name=name,
                                                               kshape=[1,3,3,in_channels,out_channels]),
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
        x=GroupNorm(x)
        x=tf.nn.relu(x)
        x=convT(name+'_T',x,self.planes,self.planes)
        x=GroupNorm(x)
        x=tf.nn.relu(x)
        return x
    
    def ST_B(self,name,x):
        tmp_x=convS(name+'_S',x,self.planes,self.planes)
        tmp_x=GroupNorm(tmp_x)
        tmp_x=tf.nn.relu(tmp_x)
        x=convT(name+'_T',x,self.planes,self.planes)
        x=GroupNorm(x)
        x=tf.nn.relu(x)
        return x+tmp_x
    
    def ST_C(self,name,x):
        x=convS(name+'_S',x,self.planes,self.planes)
        x=GroupNorm(x)
        x=tf.nn.relu(x)
        tmp_x=convT(name+'_T',x,self.planes,self.planes)
        tmp_x=GroupNorm(tmp_x)
        tmp_x=tf.nn.relu(tmp_x)
        return x+tmp_x
    
    def infer(self):
        residual=self.X_input
        if self.n_s<self.depth_3d:
            out=tf.nn.conv3d(self.X_input,get_conv_weight('conv3_{}_1'.format(self.id),[1,1,1,self.inplanes,self.planes]),
                             strides=self.stride_p,padding='SAME')
            out=GroupNorm(out)
            
        else:
            param=self.stride_p
            param.pop(1)
            out=tf.nn.conv2d(self.X_input,get_conv_weight('conv2_{}_1'.format(self.id),[1,1,self.inplanes,self.planes]),
                             strides=param,padding='SAME')
            out=GroupNorm(out)
    
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
            out=GroupNorm(out)
            out=tf.nn.relu(out)

        if self.n_s<self.depth_3d:
            out=tf.nn.conv3d(out,get_conv_weight('conv3_{}_3'.format(self.id),[1,1,1,self.planes,self.planes*BLOCK_EXPANSION]),
                             strides=[1,1,1,1,1],padding='SAME')
            out=GroupNorm(out)
        else:
            out=tf.nn.conv2d(out,get_conv_weight('conv2_{}_3'.format(self.id),[1,1,self.planes,self.planes*BLOCK_EXPANSION]),
                             strides=[1,1,1,1],padding='SAME')
            out=GroupNorm(out)
           
        if len(self.downsample)==1:
            residual=tf.nn.conv2d(residual,get_conv_weight('dw2d_{}'.format(self.id),[1,1,self.inplanes,self.planes*BLOCK_EXPANSION]),
                                  strides=[1,2,2,1],padding='SAME')
            residual=GroupNorm(residual)
        elif len(self.downsample)==2:
            residual=tf.nn.conv3d(residual,get_conv_weight('dw3d_{}'.format(self.id),[1,1,1,self.inplanes,self.planes*BLOCK_EXPANSION]),
                                  strides=self.downsample[1],padding='SAME')
            residual=GroupNorm(residual)
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

# done 4000 8000
def inference_p3d(_X,_dropout,batch_size=2, training=True):
    cnt=0
    # 16 112 112 3
    conv1_custom=tf.nn.conv3d(_X,get_conv_weight('firstconv1',[1,7,7,3,64]),strides=[1,1,2,2,1],padding='SAME')
    conv1_custom_bn=GroupNorm(conv1_custom)
    conv1_custom_bn_relu=tf.nn.relu(conv1_custom_bn)
    # 16 56 56 64
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
    deconv1_bn = GroupNorm(deconv1)
    deconv1_re = tf.nn.relu(deconv1_bn)
    # deconv1_concat = tf.concat([deconv1_re, pool3], axis=-1)
    # 2 14 14
    deconv2 = tf.layers.conv3d_transpose(deconv1_re, 256, [2, 3, 3], [2, 2, 2], 'same')
    deconv2_bn = GroupNorm(deconv2)
    deconv2_re = tf.nn.relu(deconv2_bn)
    # deconv2_concat = tf.concat([deconv2_re, pool2], axis=-1)
    # 4 28 28
    deconv3 = tf.layers.conv3d_transpose(deconv2_re, 128, 3, [2, 2, 2], 'same')
    deconv3_bn = GroupNorm(deconv3)
    deconv3_re = tf.nn.relu(deconv3_bn)
    deconv3_drop = tf.layers.dropout(deconv3_re, rate=_dropout, training=training)
    # 8 56 56
    results = tf.layers.conv3d_transpose(deconv3_drop, 1, 3, [2, 2, 2], 'same')
    # convlution
    return results

def inference_p3d_sa(_X,_dropout,batch_size=2, training=True):
    cnt=0
    # 16 112 112 3
    conv1_custom=tf.nn.conv3d(_X,get_conv_weight('firstconv1',[1,7,7,3,64]),strides=[1,1,2,2,1],padding='SAME')
    conv1_custom_bn=GroupNorm(conv1_custom)
    conv1_custom_bn_relu=tf.nn.relu(conv1_custom_bn)
    # 16 56 56 64
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

    ## attention 
    sa_1 = attention(pool4, 1024, 'sa_1', False)
    ###
    ### Deconvoltuion
    ###
    deconv1 = tf.layers.conv3d_transpose(sa_1, 512, [1, 3, 3], [2, 2, 2], 'same')
    deconv1_bn = GroupNorm(deconv1)
    deconv1_re = tf.nn.relu(deconv1_bn)
    ## attention 
    # 2 14 14
    deconv2 = tf.layers.conv3d_transpose(deconv1_re, 256, [2, 3, 3], [2, 2, 2], 'same')
    deconv2_bn = GroupNorm(deconv2)
    deconv2_re = tf.nn.relu(deconv2_bn)
    ## attention 
    # 4 28 28
    deconv3 = tf.layers.conv3d_transpose(deconv2_re, 128, 3, [2, 2, 2], 'same')
    deconv3_bn = GroupNorm(deconv3)
    deconv3_re = tf.nn.relu(deconv3_bn)
    # sa_4 = attention(deconv3_re, 128, 'sa_4', False)
    sa_4_drop = tf.layers.dropout(deconv3_re, rate=_dropout, training=training)
    # 8 56 56
    results = tf.layers.conv3d_transpose(sa_4_drop, 1, 3, [2, 2, 2], 'same')
    # 16 112 112
    return results

def inference_p3d_rsa(_X,_dropout,batch_size=2, training=True):
    cnt=0
    # 16 112 112 3
    conv1_custom=tf.nn.conv3d(_X,get_conv_weight('firstconv1',[1,7,7,3,64]),strides=[1,1,2,2,1],padding='SAME')
    conv1_custom_bn=GroupNorm(conv1_custom)
    conv1_custom_bn_relu=tf.nn.relu(conv1_custom_bn)
    # 16 56 56 64
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

    ## attention 
    sa_1 = attention(pool4, 1024, 'sa_1', False)
    ###
    ### Deconvoltuion
    ###
    deconv1 = tf.layers.conv3d_transpose(sa_1, 512, [1, 3, 3], [2, 2, 2], 'same')
    deconv1_bn = GroupNorm(deconv1)
    deconv1_re = tf.nn.relu(deconv1_bn)
    ## attention 
    sa_2 = attention(deconv1_re, 512, 'sa_2', False)
    # 2 14 14
    deconv2 = tf.layers.conv3d_transpose(sa_2, 256, [2, 3, 3], [2, 2, 2], 'same')
    deconv2_bn = GroupNorm(deconv2)
    deconv2_re = tf.nn.relu(deconv2_bn)
    ## attention 
    sa_3 = attention(deconv2_re, 256, 'sa_3', False)
    # 4 28 28
    deconv3 = tf.layers.conv3d_transpose(sa_3, 128, 3, [2, 2, 2], 'same')
    deconv3_bn = GroupNorm(deconv3)
    deconv3_re = tf.nn.relu(deconv3_bn)
    # sa_4 = attention(deconv3_re, 128, 'sa_4', False)
    sa_4_drop = tf.layers.dropout(deconv3_re, rate=_dropout, training=training)
    # 8 56 56
    results = tf.layers.conv3d_transpose(sa_4_drop, 1, 3, [2, 2, 2], 'same')
    # 16 112 112
    return results



def inference_p3d_concat(_X,_dropout,batch_size=2, training=True):
    cnt=0
    # 16 112 112 3
    conv1_custom=tf.nn.conv3d(_X,get_conv_weight('firstconv1',[1,7,7,3,64]),strides=[1,1,2,2,1],padding='SAME')
    conv1_custom_bn=GroupNorm(conv1_custom)
    conv1_custom_bn_relu=tf.nn.relu(conv1_custom_bn)
    # 16 56 56 64
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
    # from 2 14 14 to 4 28 28 512
    deconv_pool3 = tf.layers.conv3d_transpose(pool3, 512, 3, [2, 2, 2], 'same',name='deconv_pool3')
    deconv_pool3_gn = GNReLU(deconv_pool3, 'deconv_pool3_gn')

    b3=make_block(pool3,256,36,512,cnt,stride=2)
    res3=b3.infer()
    # 2 7 7 1024
    cnt=b3.cnt
    pool4=tf.nn.max_pool3d(res3,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    # 1 7 7 1024
    # from 1 7 7 to 4 28 28 1024
    deconv_pool4 = tf.layers.conv3d_transpose(pool4, 1024, 3, [4, 4, 4], 'same',name='deconv_pool4')
    deconv_pool4_gn = GNReLU(deconv_pool4, 'deconv_pool4_gn')
    ###
    ### Easy Upsampling
    ###
    concatenator = tf.concat([deconv_pool3_gn, deconv_pool4_gn, pool2], axis=-1, name='concatenator')
    conv_concat = tf.layers.conv3d(concatenator, 1024, 3, 1, 'same', name='conv_concat')
    conv_concat = GNReLU(conv_concat)
    deconv1_revise = tf.layers.conv3d_transpose(conv_concat, 256, 3, 2, 'same', name='deconv_revise')
    deconv1_revise = GNReLU(deconv1_revise, 'deconv_revise')
    deconv1_revise = tf.layers.dropout(deconv1_revise, rate=_dropout, training=training)
    results = tf.layers.conv3d_transpose(deconv1_revise, 1, 3, 2, 'same', name='predict_revise')
    return results

def inference_p3d_sa_concat(_X,_dropout,batch_size=2, training=True):
    cnt=0
    # 16 112 112 3
    conv1_custom=tf.nn.conv3d(_X,get_conv_weight('firstconv1',[1,7,7,3,64]),strides=[1,1,2,2,1],padding='SAME')
    conv1_custom_bn=GroupNorm(conv1_custom)
    conv1_custom_bn_relu=tf.nn.relu(conv1_custom_bn)
    # 16 56 56 64
    pool1=tf.nn.max_pool3d(conv1_custom_bn_relu,[1,2,3,3,1],strides=[1,2,2,2,1],padding='SAME')
    # 8 28 28 64
    b1=make_block(pool1,64,3,64,cnt)
    res1=b1.infer()
    # 8 28 28 256
    cnt=b1.cnt
    pool2=tf.nn.max_pool3d(res1,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    pool2_sa = attention(pool2, 256, 'pool2_sa', False)

    # 4 28 28 256
    b2=make_block(pool2,128,8,256,cnt,stride=2)
    res2=b2.infer()
    # 4 14 14 512 
    cnt=b2.cnt
    pool3=tf.nn.max_pool3d(res2,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    ## attention 
    pool3_sa = attention(pool3, 512, 'pool3_sa', False)
    # 2 14 14 512
    # from 2 14 14 to 4 28 28 512
    deconv_pool3 = tf.layers.conv3d_transpose(pool3_sa, 512, 3, [2, 2, 2], 'same',name='deconv_pool3')
    deconv_pool3_gn = GNReLU(deconv_pool3, 'deconv_pool3_gn')

    b3=make_block(pool3,256,36,512,cnt,stride=2)
    res3=b3.infer()
    # 2 7 7 1024
    cnt=b3.cnt
    pool4=tf.nn.max_pool3d(res3,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    ## attention 
    pool4_sa = attention(pool4, 1024, 'pool4_sa', False)
    # 1 7 7 1024
    # from 1 7 7 to 4 28 28 1024
    deconv_pool4 = tf.layers.conv3d_transpose(pool4_sa, 1024, 3, [4, 4, 4], 'same',name='deconv_pool4')
    deconv_pool4_gn = GNReLU(deconv_pool4, 'deconv_pool4_gn')
    ###
    ### Easy Upsampling
    ###
    concatenator = tf.concat([deconv_pool3_gn, deconv_pool4_gn, pool2_sa], axis=-1, name='concatenator')
    conv_concat = tf.layers.conv3d(concatenator, 1024, 3, 1, 'same', name='conv_concat')
    conv_concat = GNReLU(conv_concat)
    deconv1_revise = tf.layers.conv3d_transpose(conv_concat, 256, 3, 2, 'same', name='deconv_revise')
    deconv1_revise = GNReLU(deconv1_revise, 'deconv_revise')
    deconv1_revise = tf.layers.dropout(deconv1_revise, rate=_dropout, training=training)
    results = tf.layers.conv3d_transpose(deconv1_revise, 1, 3, 2, 'same', name='predict_revise')
    return results


def inference_p3d_sa_concat_2(_X,_dropout,batch_size=2, training=True):
    cnt=0
    # 16 112 112 3
    conv1_custom=tf.nn.conv3d(_X,get_conv_weight('firstconv1',[1,7,7,3,64]),strides=[1,1,2,2,1],padding='SAME')
    conv1_custom_bn=GroupNorm(conv1_custom)
    conv1_custom_bn_relu=tf.nn.relu(conv1_custom_bn)
    # 16 56 56 64
    pool1=tf.nn.max_pool3d(conv1_custom_bn_relu,[1,2,3,3,1],strides=[1,2,2,2,1],padding='SAME')
    # 8 28 28 64
    b1=make_block(pool1,64,3,64,cnt)
    res1=b1.infer()
    # 8 28 28 256
    cnt=b1.cnt
    pool2=tf.nn.max_pool3d(res1,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    pool2_sa = attention(pool2, 256, 'pool2_sa', False)

    # 4 28 28 256
    b2=make_block(pool2,128,8,256,cnt,stride=2)
    res2=b2.infer()
    # 4 14 14 512 
    cnt=b2.cnt
    pool3=tf.nn.max_pool3d(res2,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    # 2 14 14 512
    # from 2 14 14 to 4 28 28 512
    deconv_pool3 = tf.layers.conv3d_transpose(pool3, 512, 3, [2, 2, 2], 'same',name='deconv_pool3')
    deconv_pool3_gn = GNReLU(deconv_pool3, 'deconv_pool3_gn')
    ## attention 
    deconv_pool3_gn_sa = attention(deconv_pool3_gn, 512, 'deconv_pool3_gn_sa', False)

    b3=make_block(pool3,256,36,512,cnt,stride=2)
    res3=b3.infer()
    # 2 7 7 1024
    cnt=b3.cnt
    pool4=tf.nn.max_pool3d(res3,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    ## attention 
    # 1 7 7 1024
    # from 1 7 7 to 4 28 28 1024
    deconv_pool4 = tf.layers.conv3d_transpose(pool4, 512, 3, [4, 4, 4], 'same',name='deconv_pool4')
    deconv_pool4_gn = GNReLU(deconv_pool4, 'deconv_pool4_gn')
    deconv_pool4_gn_sa = attention(deconv_pool4_gn, 512, 'deconv_pool4_gn_sa', False)
    ###
    ### Easy Upsampling
    ###
    concatenator = tf.concat([pool2_sa, deconv_pool3_gn_sa, deconv_pool4_gn_sa], axis=-1, name='concatenator')
    conv_concat = tf.layers.conv3d(concatenator, 512, 3, 1, 'same', name='conv_concat')
    conv_concat = GNReLU(conv_concat)
    deconv1_revise = tf.layers.conv3d_transpose(conv_concat, 128, 3, 2, 'same', name='deconv_revise')
    deconv1_revise = GNReLU(deconv1_revise, 'deconv_revise')
    deconv1_revise = tf.layers.dropout(deconv1_revise, rate=_dropout, training=training)
    results = tf.layers.conv3d_transpose(deconv1_revise, 1, 3, 2, 'same', name='predict_revise')
    return results


def inference_p3d_sa_decoder_block(_X,_dropout,batch_size=2, training=True):
    cnt=0
    # 16 112 112 3
    conv1_custom=tf.nn.conv3d(_X,get_conv_weight('firstconv1',[1,7,7,3,64]),strides=[1,1,2,2,1],padding='SAME')
    conv1_custom_bn=GroupNorm(conv1_custom)
    conv1_custom_bn_relu=tf.nn.relu(conv1_custom_bn)
    # 16 56 56 64
    pool1=tf.nn.max_pool3d(conv1_custom_bn_relu,[1,2,3,3,1],strides=[1,2,2,2,1],padding='SAME')
    # 8 28 28 64
    b1=make_block(pool1,64,3,64,cnt)
    res1=b1.infer()
    # 8 28 28 256
    cnt=b1.cnt
    pool2=tf.nn.max_pool3d(res1,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    pool2_sa = attention(pool2, 256, 'pool2_sa', False)
    # 4 28 28 256
    b2=make_block(pool2,128,8,256,cnt,stride=2)
    res2=b2.infer()
    # 4 14 14 512 
    cnt=b2.cnt
    pool3=tf.nn.max_pool3d(res2,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    pool3_sa = attention(pool3, 512, 'pool3_sa', False)
    # 2 14 14 512
    deconv_pool3 = tf.layers.conv3d_transpose(pool3_sa, 512, [2, 3, 3], [2, 2, 2],'same',  name='deconv_pool3')
    deconv_pool3_gn = GNReLU(deconv_pool3, 'deconv_pool3_gn')
    b3=make_block(pool3,256,36,512,cnt,stride=2)
    res3=b3.infer()
    # 2 7 7 1024
    cnt=b3.cnt
    pool4=tf.nn.max_pool3d(res3,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    pool4_sa = attention(pool4, 1024, 'pool4_sa', False)
    # from 1 7 7 to 4 28 28 1024
    deconv_pool4 = tf.layers.conv3d_transpose(pool4_sa, 1024, 3, [4, 4, 4], 'same',name='deconv_pool4')
    deconv_pool4_gn = GNReLU(deconv_pool4, 'deconv_pool4_gn')
    ###
    ### Concat
    ###
    concatenator = tf.concat([deconv_pool3_gn, deconv_pool4_gn, pool2_sa], axis=-1, name='concatenator')
    conv_concat = tf.layers.conv3d(concatenator, 1024, 3, 1, 'same', name='conv_concat')
    conv_concat = GNReLU(conv_concat)
    # Decoder Block 1
    decoder1_conv1 = tf.layers.conv3d(conv_concat, 256, 3, 1,'same', name='decoder1_conv1')
    decoder1_conv1 = GNReLU(decoder1_conv1, 'decoder1_conv1_gn')
    decoder1_deconv = tf.layers.conv3d_transpose(decoder1_conv1, 256, 3, 2, 'same', name='decoder1_deconv')
    decoder1_deconv = GNReLU(decoder1_deconv, 'decoder1_deconv_gn')
    decoder1_conv2 = tf.layers.conv3d(decoder1_deconv, 128, 3, 1,'same', name='decoder1_conv2')
    decoder1_conv2 = GNReLU(decoder1_conv2, 'decoder1_conv2_gn')

    # Decoder Block 2
    decoder2_conv1 = tf.layers.conv3d(decoder1_conv2, 32, 3, 1, 'same', name='decoder2_conv1')
    decoder2_conv1 = GNReLU(decoder2_conv1, 'decoder2_conv1_gn')
    decoder2_deconv = tf.layers.conv3d_transpose(decoder2_conv1, 32, 3, 2, 'same', name='decoder2_deconv')
    decoder2_deconv = GNReLU(decoder2_deconv, 'decoder2_deconv_gn')
    decoder2_conv2 = tf.layers.conv3d(decoder2_deconv, 16, 3, 1, 'same',name='decoder2_conv2')
    decoder2_conv2 = GNReLU(decoder2_conv2, 'decoder2_conv2_gn')
    # Final Conv
    
    final_conv = tf.layers.dropout(decoder2_conv2, _dropout, training=training)
    results = tf.layers.conv3d(final_conv, 1, 3, 1, 'same',name='results')

    return results

