# coding=utf-8
import numpy as np
import tensorflow as tf
import sys
import os
import six
a=eval(sys.argv[1])
a=str(a)
b=eval(sys.argv[2])

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    if 'data' in dict:
        dict['data'] = dict['data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3) / 256.

    return dict

def load_data_one(f):
    batch = unpickle(f)
    data = batch['data']
    labels = batch['labels']
    print "Loading %s: %d" % (f, len(data))
    return data, labels

def load_data(files, data_dir, label_count):
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([ [ float(i == label) for i in xrange(label_count) ] for label in labels ])
    return data, labels


def my_load(file, data_dir):
    batch = np.load(data_dir + '/' + file)
    imageData = np.array([np.array(item['feature']).reshape(128*128*3)/256. for item in batch])
    labels = np.array([np.array(item['label']) for item in batch])
    return imageData,labels


def run_in_batch_avg(session, tensors, batch_placeholders, feed_dict={}, batch_size=200):                              
    res = [ 0 ] * len(tensors)
    batch_tensors = [ (placeholder, feed_dict[ placeholder ]) for placeholder in batch_placeholders ]                    
    total_size = len(batch_tensors[0][1])                                                                                
    batch_count = (total_size + batch_size - 1) / batch_size                                                             
    for batch_idx in xrange(batch_count):                                                                                
        current_batch_size = None                                                                                          
        for (placeholder, tensor) in batch_tensors:                                                                        
            batch_tensor = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]                                         
            current_batch_size = len(batch_tensor)                                                                           
            feed_dict[placeholder] = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]                               
        tmp = session.run(tensors, feed_dict=feed_dict)                                                                    
        res = [ r + t * current_batch_size for (r, t) in zip(res, tmp) ]                                                   
    return [ r / float(total_size) for r in res ]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
    W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
    conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
    if with_bias:
        return conv + bias_variable([ out_features ])
    return conv
  
def conv2dReduce(input, in_features, out_features, kernel_size, with_bias=False):
    W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
    conv = tf.nn.conv2d(input, W, [ 1, 2, 2, 1 ], padding='VALID')
    if with_bias:
        return conv + bias_variable([ out_features ])
    return conv
def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob):
    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.nn.relu(current)
    current = conv2d(current, in_features, out_features, kernel_size)
    current = tf.nn.dropout(current, keep_prob)
    return current

def block(input, layers, in_features, growth, is_training, keep_prob):
    current = input
    features = in_features
    for idx in xrange(layers):
        tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob)
        current = tf.concat((current, tmp), axis=3)
        features += growth
    return current, features

def avg_pool(input, s):
    return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')

def _bottleneck_residual(x, in_filter, out_filter, stride,is_training):

    with tf.variable_scope('residual_bn_relu'):
        # 获取残差直连
        orig_x = x
        # 后做BN和ReLU激活
        x = tf.contrib.layers.batch_norm(x, scale=True, is_training=is_training, updates_collections=None)
        x = tf.nn.relu(x,)

    # 第1子层
    with tf.variable_scope('sub1'):
        # 1x1卷积，使用输入步长，通道数(in_filter -> out_filter/4)
        if stride == 1:
            x = conv2d(x, in_filter, out_filter / 4, 1, )
        elif stride == 2:
            x = conv2dReduce( x, in_filter, out_filter/4, 1,)

    # 第2子层
    with tf.variable_scope('sub2'):
          # BN和ReLU激活
          x = tf.contrib.layers.batch_norm(x, scale=True, is_training=is_training, updates_collections=None)
          x = tf.nn.relu(x, )
          # 3x3卷积，步长为1，通道数不变(out_filter/4)
          x = conv2d( x,  out_filter/4, out_filter/4, 3)

    # 第3子层
    with tf.variable_scope('sub3'):
          # BN和ReLU激活
          x = tf.contrib.layers.batch_norm(x, scale=True, is_training=is_training, updates_collections=None)
          x = tf.nn.relu(x, )
          # 1x1卷积，步长为1，通道数不变(out_filter/4 -> out_filter)
          x = conv2d( x, out_filter/4, out_filter, 1)



    # 合并残差层
    with tf.variable_scope('sub_add'):
      # 当通道数有变化时
          if in_filter != out_filter:
            # 1x1卷积，使用输入步长，通道数(in_filter -> out_filter)
	      if stride == 1:
                  orig_x = conv2d( orig_x,  in_filter, out_filter, 1)
              if stride == 2:
                  orig_x = conv2dReduce( orig_x,  in_filter, out_filter, 1)
              x += orig_x
	    
          # 合并残差
          else:
              x += orig_x
        #x=tf.contrib.layers.instance_norm(x)


    tf.logging.info('image after unit %s', x.get_shape())
    return x


def run_model(data, image_dim, label_count, depth):
    weight_decay = 1e-4
    # layers = (depth - 6) / 6
    layers = b
    print layers
    graph = tf.Graph()
    with graph.as_default():
        xs = tf.placeholder("float", shape=[None, image_dim],name='xs')
        ys = tf.placeholder("float", shape=[None, label_count],name='ys')
        lr = tf.placeholder("float", shape=[],name='lr')
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        is_training = tf.placeholder("bool", shape=[],name='is_training')

	filters = [16, 64, 128, 256, 512]
        strides = [1, 2, 2, 2]
        current = tf.reshape(xs, [ -1, 128, 128, 3 ])

        current = conv2d(current,3,16,2)
#         current = conv2dReduce(current, 3, 16, 3)
#         current = conv2dReduce(current,3,16,2)
        #100

        #第一组
        with tf.variable_scope('unit_1_0'):
            current = _bottleneck_residual(current, filters[0], filters[1],
                         strides[0],is_training)
        for i in six.moves.range(1, 3):
            with tf.variable_scope('unit_1_%d' % i):
                current = _bottleneck_residual(current, filters[1], filters[1], 1,is_training)
        #第二组
        with tf.variable_scope('unit_2_0'):
            current = _bottleneck_residual(current, filters[1], filters[2],
                         strides[1],is_training)
        for i in six.moves.range(1, 4):
            with tf.variable_scope('unit_2_%d' % i):
                current = _bottleneck_residual(current, filters[2], filters[2], 1,is_training)
        #第三组
        with tf.variable_scope('unit_3_0'):
            current = _bottleneck_residual(current, filters[2], filters[3],strides[2],is_training)
        for i in six.moves.range(1, 6):
            with tf.variable_scope('unit_3_%d' % i):
                current = _bottleneck_residual(current, filters[3], filters[3], 1,is_training)
        #第四组
        with tf.variable_scope('unit_4_0'):
            current = _bottleneck_residual(current, filters[3], filters[4], strides[3],is_training)
                       

        for i in six.moves.range(1, 3):
            with tf.variable_scope('unit_4_%d' % i):
                current = _bottleneck_residual(current, filters[4], filters[4],1,is_training)
        features = filters[4]
	current = avg_pool(current, 4)


        current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
        current = tf.nn.relu(current,name='relu')
        current = avg_pool(current, 4)
        final_dim = features
    
    
#         print current.get_shape
#         print ys.get_shape
#         current = tf.concat([current,tf.reshape(zs,[-1,1,1,21])],3)
    
        current = tf.reshape(current, [ -1, final_dim ],name='finalCurrent')
        Wfc = weight_variable([ final_dim, label_count ])
        bfc = bias_variable([ label_count ])
        ys_ = tf.nn.softmax( tf.matmul(current, Wfc) + bfc,name='ys_' )
        tf.add_to_collection("ys_",ys_)

        ys_shape = ys.shape
        _ys_shape = ys_.shape
#         print ys_shape
#         print _ys_shape
        cross_entropy = -tf.reduce_mean(ys * tf.log(ys_ + 1e-12))
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(cross_entropy + l2 * weight_decay)
        correct_prediction = tf.equal(tf.argmax(ys_, 1), tf.argmax(ys, 1),name='correct_prediction')
        tf.add_to_collection("correct_prediction",correct_prediction);
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session(graph=graph) as session:
        # writer = tf.summary.FileWriter('trained/', session.graph)
        batch_size = 32
        learning_rate = 0.01
        saver = tf.train.Saver()
        # saver.restore(session, tf.train.latest_checkpoint('trained/'))
        session.run(tf.global_variables_initializer())
        train_imageData, train_labels = data['train_imageData'], data['train_labels']
        batch_count = len(train_imageData) / batch_size
        batches_imageData = np.split(train_imageData[:batch_count * batch_size], batch_count)
        batches_labels = np.split(train_labels[:batch_count * batch_size], batch_count)
        print "Batch per epoch: ", batch_count

	#
	desPath = 'aoeiutafotetitu/fold'+a+'/layer'+str(b)
        if not os.path.exists(desPath):
            os.makedirs(desPath)
        f=open('aoeiutafotetitu/result_Fold'+a+'_layer'+str(b)+'.txt','wb')

        for epoch in xrange(1, 1+400):
            if epoch == 50: learning_rate = 0.001
            if epoch == 250: learning_rate = 0.0001
            if epoch == 450:learning_rate = 0.00001
            if epoch == 750:learning_rate = 0.000001
#           if epoch == 950:learning_rate = 0.000001
            for batch_idx in xrange(batch_count):
                xs_, ys_ = batches_imageData[batch_idx], batches_labels[batch_idx]
                batch_res = session.run([ train_step, cross_entropy, accuracy ],
                                        feed_dict = { xs: xs_, ys: ys_, lr: learning_rate, is_training: True, keep_prob: 0.85 })
        # if batch_idx % 10 == 0: print epoch, batch_idx, batch_res[1:]
            if epoch%50==0:
                save_path = saver.save(session, desPath+ '/denseNet_%d.ckpt' % epoch)
            test_results = run_in_batch_avg(session, [ cross_entropy, accuracy], [ xs, ys ],feed_dict = { xs: data['test_imageData'], ys: data['test_labels'],is_training: False, keep_prob: 1. })
            print epoch, batch_res[1:], test_results
	    f.write(str(epoch)+'\t'+str(test_results[0])+'\t'+str(test_results[1])+'\n')
        # save_path = saver.save(session, 'trained/denseNet_%d.ckpt' % epoch)
        f.close()

def run():
    data_dir = 'data'
    image_size = 128
    image_dim = image_size * image_size * 3
    #label_names = ['benign', 'vious']
    #label_count = len(label_names)
    label_count = 10

    train_imageData,train_labels = my_load('aoeiu_24/Fold'+a+'_train.npy', '.')
    test_imageData,test_labels = my_load('aoeiu_24/Fold'+a+'_test.npy', '.')
    print "Train:", np.shape(train_imageData), np.shape(train_labels)
    print "Test:", np.shape(test_imageData), np.shape(test_labels)
    data = { 'train_imageData': train_imageData,
            'train_labels': train_labels,
            'test_imageData': test_imageData,
            'test_labels': test_labels }

    run_model(data, image_dim, label_count, 66)

run()
