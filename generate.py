# Use a trained Image Transform Net to generate
# a style transferred image with a specific style

import tensorflow as tf
import numpy as np

from fusion_l1norm import L1_norm
from densefuse_net import DenseFuseNet
from utils import get_images, save_images, get_train_images

def generate(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index, type='addition', output_path=None):

    if type == 'addition':
        print('addition')
        outputs = _handler(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index, output_path=output_path)
    elif type == 'l1':
        print('l1')
        outputs = _handler_l1(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index,
                       output_path=output_path)

    return list(outputs)


def _handler(content_name, style_name, model_path, model_pre_path, ssim_weight, index, output_path=None):
    infrared_path = content_name
    visible_path = style_name

    content_img = get_train_images(infrared_path, flag=False)
    style_img   = get_train_images(visible_path, flag=False)
    dimension = content_img.shape

    content_img = content_img.reshape([1, dimension[0], dimension[1], dimension[2]])
    style_img   = style_img.reshape([1, dimension[0], dimension[1], dimension[2]])

    content_img = np.transpose(content_img, (0, 2, 1, 3))
    style_img = np.transpose(style_img, (0, 2, 1, 3))
    print('content_img shape final:', content_img.shape)

    with tf.Graph().as_default(), tf.Session() as sess:

        # build the dataflow graph
        content = tf.placeholder(
            tf.float32, shape=content_img.shape, name='content')
        style = tf.placeholder(
            tf.float32, shape=style_img.shape, name='style')

        dfn = DenseFuseNet(model_pre_path)

        output_image = dfn.transform_addition(content,style)
        # output_image = dfn.transform_recons(style)
        # output_image = dfn.transform_recons(content)

        # restore the trained model and run the style transferring
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        output = sess.run(output_image, feed_dict={content: content_img, style: style_img})
        save_images(infrared_path, output, output_path,
                    prefix='fused' + str(index), suffix='_densefuse_addition_'+str(ssim_weight))

    return output


def _handler_l1(content_name, style_name, model_path, model_pre_path, ssim_weight, index, output_path=None):
    infrared_path = content_name
    visible_path = style_name

    content_img = get_train_images(infrared_path, flag=False)
    style_img   = get_train_images(visible_path, flag=False)
    dimension = content_img.shape

    content_img = content_img.reshape([1, dimension[0], dimension[1], dimension[2]])
    style_img   = style_img.reshape([1, dimension[0], dimension[1], dimension[2]])

    content_img = np.transpose(content_img, (0, 2, 1, 3))
    style_img = np.transpose(style_img, (0, 2, 1, 3))
    print('content_img shape final:', content_img.shape)

    with tf.Graph().as_default(), tf.Session() as sess:

        # build the dataflow graph
        content = tf.placeholder(
            tf.float32, shape=content_img.shape, name='content')
        style = tf.placeholder(
            tf.float32, shape=style_img.shape, name='style')

        dfn = DenseFuseNet(model_pre_path)

        enc_c = dfn.transform_encoder(content)
        enc_s = dfn.transform_encoder(style)

        target = tf.placeholder(
            tf.float32, shape=enc_c.shape, name='target')

        output_image = dfn.transform_decoder(target)

        # restore the trained model and run the style transferring
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        enc_c_temp, enc_s_temp = sess.run([enc_c, enc_s], feed_dict={content: content_img, style: style_img})
        feature = L1_norm(enc_c_temp, enc_s_temp)

        output = sess.run(output_image, feed_dict={target: feature})
        save_images(infrared_path, output, output_path,
                    prefix='fused' + str(index), suffix='_densefuse_l1norm_'+str(ssim_weight))

    return output

