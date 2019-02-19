# coding=utf-8
import numpy as np
import tensorflow as tf
import sys
import os
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
    csvData = np.array([np.array(item['csv']) for item in batch])
    labels = np.array([np.array(item['label']) for item in batch])
    return imageData,csvData,labels


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
def _conv(name, x, filter_size, in_filters, out_filters, strides):
    with tf.variable_scope(name):
       n = filter_size * filter_size * out_filters
     
       kernel = tf.get_variable(
              'DW',
              [filter_size, filter_size, in_filters, out_filters],
              tf.float32,
              initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
       #
       return tf.nn.conv2d(x, kernel, strides, padding='SAME')
def _bottleneck_residual(x, in_filter, out_filter, stride,
                         activate_before_residual=False):
    # 
    if activate_before_residual:
        with tf.variable_scope('common_bn_relu'):
            # 
            x =tf.contrib.layers.batch_norm(x, scale=True, is_training=is_training, updates_collections=None)
            x = tf.nn.relu(x)
            # 
            orig_x = x
    else:
        with tf.variable_scope('residual_bn_relu'):
            # 
            orig_x = x
            #
            x = tf.contrib.layers.batch_norm(x, scale=True, is_training=is_training, updates_collections=None)
            x = tf.nn.relu(x)

    # 
    with tf.variable_scope('sub1'):
        # 
        x = _conv('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])

    #
    with tf.variable_scope('sub2'):
        # 
        x = tf.contrib.layers.batch_norm(x, scale=True, is_training=is_training, updates_collections=None)
        x = tf.nn.relu(x)
        # 
        x = _conv('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])

    # 
    with tf.variable_scope('sub3'):
        #
        x = tf.contrib.layers.batch_norm(x, scale=True, is_training=is_training, updates_collections=None)
        x = tf.nn.relu(x)
        # 
        x = _conv('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1])

    # 
    with tf.variable_scope('sub_add'):
        # 
        if in_filter != out_filter:
            # 
            orig_x = _conv('project', orig_x, 1, in_filter, out_filter, stride)

        # 
        x += orig_x

    tf.logging.info('image after unit %s', x.get_shape())
    return x
def _fully_connected(self, x, out_dim):
    # 
    x = tf.reshape(x, [self.hps.batch_size, -1])
    # 
    w = tf.get_variable('DW', [x.get_shape()[1], out_dim],
                        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    # 
    b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
    # 
    return tf.nn.xw_plus_b(x, w, b)
def avg_pool(input, s):
    return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')
def _global_avg_pool( x):
    assert x.get_shape().ndims == 4
    # 
    return tf.reduce_mean(x, [1, 2])
def _stride_arr(stride):
    return [1, stride, stride, 1]
def run_model(data, image_dim, label_count, csv_dim, depth):
    weight_decay = 1e-4
    # layers = (depth - 6) / 6
    layers = 11
    print layers
    graph = tf.Graph()
    with graph.as_default():
        xs = tf.placeholder("float", shape=[None, image_dim],name='xs')
        ys = tf.placeholder("float", shape=[None, label_count],name='ys')
        zs = tf.placeholder("float", shape=[None, csv_dim],name='zs')
        lr = tf.placeholder("float", shape=[],name='lr')
        keep_prob = tf.placeholder(tf.float32,name='keep_proc')
        is_training = tf.placeholder("bool", shape=[],name='is_training')
        filters = [16, 64, 128, 256]

        current = tf.reshape(xs, [ -1, 128, 128, 3 ])

        current = _conv('init_conv', x, 3, 3, 16, _stride_arr(1))
#         current = conv2dReduce(current, 3, 16, 3)
#         current = conv2dReduce(current,3,16,2)
        #100
        strides = [1, 2, 2]
        # 
        activate_before_residual = [True, False, False]
        with tf.variable_scope('unit_1_0'):
            current = _bottleneck_residual(current, filters[0], filters[1],
                         _stride_arr(strides[0]),
                         activate_before_residual[0])
        for i in six.moves.range(1, 4):
            with tf.variable_scope('unit_1_%d' % i):
                x = _bottleneck_residual(x, filters[1], filters[1], _stride_arr(1), False)

        with tf.variable_scope('unit_2_0'):
            x = _bottleneck_residual(x, filters[1], filters[2],
                         _stride_arr(strides[1]),
                         activate_before_residual[1])
        for i in six.moves.range(1, 4):
            with tf.variable_scope('unit_2_%d' % i):
                x = _bottleneck_residual(x, filters[2], filters[2], _stride_arr(1), False)

        with tf.variable_scope('unit_3_0'):
            x = _bottleneck_residual(x, filters[2], filters[3], _stride_arr(strides[2]),
                         activate_before_residual[2])
        for i in six.moves.range(1, 4):
            with tf.variable_scope('unit_3_%d' % i):
                x = _bottleneck_residual(x, filters[3], filters[3], _stride_arr(1), False)


        current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
        current = tf.nn.relu(current,name='relu')
        current = _global_avg_pool(current)

    
    
        print current.get_shape
        print ys.get_shape

        with tf.variable_scope('logit'):
            logits = _fully_connected(x, self.hps.num_classes)
            ys_ = tf.nn.softmax(logits)

        #Wfc = weight_variable([ final_dim, label_count ])
        #bfc = bias_variable([ label_count ])
        #ys_ = tf.nn.softmax( tf.matmul(current, Wfc) + bfc ,name='ys_')
        #tf.add_to_collection("ys_",ys_)

        ys_shape = ys.shape
        _ys_shape = ys_.shape
#         print ys_shape
#         print _ys_shape
        cross_entropy = -tf.reduce_mean(ys * tf.log(ys_ + 1e-12))
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(cross_entropy + l2 * weight_decay)
        correct_prediction = tf.equal(tf.argmax(ys_, 1), tf.argmax(ys, 1),name='correct_prediction')
        tf.add_to_collection("correct_prediction",correct_prediction)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session(graph=graph) as session:
        # writer = tf.summary.FileWriter('trained/', session.graph)
        batch_size = 32
        learning_rate = 0.01
        saver = tf.train.Saver()
        # saver.restore(session, tf.train.latest_checkpoint('trained/'))
        session.run(tf.global_variables_initializer())
        train_imageData, train_csvData,train_labels = data['train_imageData'], data['train_csvData'],data['train_labels']
        batch_count = len(train_imageData) / batch_size
        batches_imageData = np.split(train_imageData[:batch_count * batch_size], batch_count)
        batches_csvData = np.split(train_csvData[:batch_count * batch_size], batch_count)
        batches_labels = np.split(train_labels[:batch_count * batch_size], batch_count)
        print "Batch per epoch: ", batch_count
        for epoch in xrange(1, 1+1000):
            if epoch == 50: learning_rate = 0.001
            if epoch == 250: learning_rate = 0.0001
            if epoch == 450:learning_rate = 0.00001
            if epoch == 750:learning_rate = 0.000001
#           if epoch == 950:learning_rate = 0.000001
            for batch_idx in xrange(batch_count):
                xs_, ys_, zs_ = batches_imageData[batch_idx], batches_labels[batch_idx],batches_csvData[batch_idx]
                batch_res = session.run([ train_step, cross_entropy, accuracy ],
                                        feed_dict = { xs: xs_, ys: ys_, zs: zs_, lr: learning_rate, is_training: True, keep_prob: 0.85 })
            if epoch % 100 == 0: 
                save_path = saver.save(session, 'trained/denseNet_%d.ckpt' % epoch)
            test_results = run_in_batch_avg(session, [ cross_entropy, accuracy], [ xs, ys, zs],feed_dict = { xs: data['test_imageData'], ys: data['test_labels'], zs: data['test_csvData'],is_training: False, keep_prob: 1. })
            print epoch, batch_res[1:], test_results
        save_path = saver.save(session, 'trained/denseNet_%d.ckpt' % epoch)
def run():
    data_dir = 'data'
    image_size = 128
    image_dim = image_size * image_size * 3
    csv_dim = 57
    #label_names = ['benign', 'vious']
    #label_count = len(label_names)
    label_count = 5

    train_imageData,train_csvData,train_labels = my_load('aoeiu12/Fold'+a+'_train.npy', '.')
    test_imageData,test_csvData,test_labels = my_load('aoeiu12/Fold'+a+'_test.npy', '.')

    print "Train:", np.shape(train_imageData),np.shape(train_csvData), np.shape(train_labels)
    print "Test:", np.shape(test_imageData), np.shape(test_csvData), np.shape(test_labels)
    data = { 'train_imageData': train_imageData,
            'train_labels': train_labels,
            'test_imageData': test_imageData,
            'test_labels': test_labels ,
            'train_csvData':train_csvData,
            'test_csvData':test_csvData}
    run_model(data, image_dim, label_count,csv_dim, 66)
def _relu(x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
run()
