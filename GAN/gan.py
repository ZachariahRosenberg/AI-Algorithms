import os
import numpy as np
from glob import glob
from matplotlib import pyplot

def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """

    real_input    = tf.placeholder(tf.float32, (None, image_width, image_height, image_channels))
    z_input       = tf.placeholder(tf.float32, (None, z_dim))
    learning_rate = tf.placeholder(tf.float32)
    return real_input, z_input, learning_rate

def generator(z, out_channel_dim, is_train=True, alpha=0.1):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """
    
    with tf.variable_scope('generator', reuse = not is_train):
        
        #input layer, fully connected
        h1 = tf.layers.dense(z, 7 * 7 * 256)
        
        #deconv layer 2, output 14x14x128
        h2   = tf.reshape(h1, (-1, 7, 7, 256))
        h2   = tf.layers.conv2d_transpose(h2, 128, 5, strides=2,padding='same')
        bn2  = tf.layers.batch_normalization(h2)
        out2 = tf.maximum(bn2, bn2*alpha)
        
        #deconv layer 3, output 28x28x3
        h3   = tf.layers.conv2d_transpose(out2, out_channel_dim, 5, strides=2,padding='same')
        bn3  = tf.layers.batch_normalization(h3)
        out3 = tf.maximum(bn3, bn3*alpha)
        
        #output
        output = tf.tanh(out3)
        
    
    return output

def discriminator(images, reuse=False, alpha=0.1):
    """
    Create the discriminator network
    :param images: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """
    # TODO: Implement Function
    with tf.variable_scope('discriminator', reuse=reuse):

        #First layer, output 14x14x64
        h1   = tf.layers.conv2d(images, 64, 5, strides=2, padding='same')
        out1 = tf.maximum(h1, h1*alpha)

        #second layer, output 7x7x128
        h2   = tf.layers.conv2d(out1, 128, 5, strides=2, padding='same')
        bn2  = tf.layers.batch_normalization(h2)
        out2 = tf.maximum(bn2, bn2*alpha)

        #third layer, output 4x4x256
        h3   = tf.layers.conv2d(out2, 256, 5, strides=2, padding='same')
        bn3  = tf.layers.batch_normalization(h3)
        out3 = tf.maximum(bn3, bn3*alpha)

        #final dense layer
        dense_input = tf.reshape(out3, (-1, 4 * 4 * 256))
        logits      = tf.layers.dense(dense_input, 1)
        out         = tf.sigmoid(logits) 
        
    return out, logits

def model_loss(input_real, input_z, out_channel_dim):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """

    g_model = generator(input_z, out_channel_dim)
    d_model_real, d_logits_real = discriminator(input_real)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)
    
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,labels=tf.ones_like(d_logits_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.zeros_like(d_logits_fake)))
    g_loss      = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.ones_like(d_logits_fake)))
    
    d_loss = d_loss_real + d_loss_fake
    
    
    return d_loss, g_loss

def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    
    d_train_vars = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')]
    g_train_vars = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')]
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_train_vars)
        g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_train_vars)
    
    return d_opt, g_opt

def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    pyplot.imshow(images_grid, cmap=cmap)
    pyplot.show()

def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """
    
    # TODO: Build Model
    image_channel = 3 if data_image_mode == 'RGB' else 1

    real_input, z_input, lr = model_inputs(data_shape[1], data_shape[2], image_channel, z_dim)
    
    d_loss, g_loss = model_loss(real_input, z_input, image_channel)
    d_opt, g_opt   = model_opt(d_loss, g_loss, lr, beta1)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            print('Starting epoch {} of {}'.format(epoch_i+1, epoch_count))
            for i, batch_images in enumerate(get_batches(batch_size)):
                # TODO: Train Model
                batch_images = batch_images * 2
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                sess.run(d_opt, feed_dict={real_input:batch_images, z_input:batch_z, lr:learning_rate})
                sess.run(g_opt, feed_dict={real_input:batch_images, z_input:batch_z, lr:learning_rate})
                
                if i+1 % 10 == 0:
                    train_loss_d = d_loss.eval({z_input: batch_z, real_input: batch_images, lr:learning_rate})
                    train_loss_g = g_loss.eval({z_input: batch_z, lr:learning_rate})
                    print('Batch {} of {}'.format(i+1, 60000//batch_size),
                          'Discriminator Loss: {:.4f}'.format(train_loss_d),
                          'Generator Loss: {:.4f}'.format(train_loss_g))
                if i+1 % 100 == 0:
                    show_generator_output(sess, 1, tf.random_uniform((1, z_dim),-1,1), image_channel, data_image_mode)

batch_size = 256
z_dim = 100
learning_rate = .001
beta1 = .5
epochs = 2

mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))
with tf.Graph().as_default():
    print(mnist_dataset.shape)
    train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
          mnist_dataset.shape, mnist_dataset.image_mode)