import numpy as np
import tensorflow as tf

import vgg16
import utils

#img1 = utils.load_image("./test_data/tiger.jpeg")
#img2 = utils.load_image("./test_data/puzzle.jpeg")
img3 = utils.load_image("./test_data/kitten2.png")

#batch1 = img1.reshape((1, 224, 224, 3))
#batch2 = img2.reshape((1, 224, 224, 3))
batch3 = img3.reshape((1, 224, 224, 3))

batch = batch3 #np.concatenate((batch1, batch2, batch3), 0)

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/GPU:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [1, 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        [prob,
         fc8, fc7, fc6,
         conv5_1, conv4_1, conv3_1,
         conv2_1, conv1_1,
         bgr, rgb_sc, rgb] = sess.run([vgg.prob,
                                       vgg.fc8,
                                       vgg.fc7,
                                       vgg.fc6,
                                       vgg.conv5_1,
                                       vgg.conv4_1,
                                       vgg.conv3_1,
                                       vgg.conv2_1,
                                       vgg.conv1_1,
                                       vgg.bgr,
                                       vgg.rgb_scaled,
                                       vgg.rgb], feed_dict=feed_dict)


        utils.print_prob(prob[0], './synset.txt')
        #utils.print_prob(prob[1], './synset.txt')        
        #utils.print_prob(prob[2], './synset.txt')
