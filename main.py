
from six.moves import urllib
import os
import sys
import cv2
import tarfile
from pathlib import Path
import re
from rotate import rotate_crop_scale
from PIL import ImageFilter

# !mkdir -p 'data'
data_dir = 'data/'
export_dir = "export/"
checkpoint_path_str = './checkpoint/'
generated_path_str = './generated/'

checkpoint_path = Path(checkpoint_path_str)
checkpoint_path.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE_W = 64
IMAGE_SIZE_L = 64

# ORIG_RESIZE = (86, 128)
ORIG_RESIZE = (43, 64)
# Original 1654/2480
# Original 827/1240

# data_url = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz'
# filename = data_url.split('/')[-1]
# filepath = data_dir + data_url.split('/')[-1]

# if not os.path.exists(filepath):
#   def progress(count, block_size, total_size):
#     sys.stdout.write('\r>> Downloading %s %.1f%%' % \
#         (filename, float(count * block_size) / total_size * 100))
#     sys.stdout.flush()
#   filepath, _ = urllib.request.urlretrieve(data_url, filepath, progress)
#   tarfile.open(filepath, 'r:gz').extractall(data_dir)

# Creating a larger dataset using data augmentation.

from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np
rnd = np.random.default_rng(seed=0)

import matplotlib.pyplot as plt
import matplotlib

class ImgData(object):
  def __init__(self, datasetid):
    self.datasetid = datasetid
    self.img_dim = (IMAGE_SIZE_W, IMAGE_SIZE_L)
    img_dir = join(data_dir, datasetid)

    img_files = [join(img_dir, f)
                for f in listdir(img_dir) if isfile(join(img_dir, f))]

    img_data_path = join(data_dir, f"{datasetid}_{IMAGE_SIZE_W}_{IMAGE_SIZE_L}.npy")
    if isfile(img_data_path):
      with open(img_data_path, 'rb') as f:
        self.data = np.load(f)
    else:
      data = []
      for f in img_files:
        if f.split('.')[-1] != 'jpg':
          continue
        img = Image.open(f)
        imgs = [rotate_crop_scale(np.asarray(img), 20.0 * (rnd.random() - 0.5)) for _ in range(10)]

        # i = 0
        # export_path = join(export_dir, self.datasetid)
        # Path(export_path).mkdir(parents=True, exist_ok=True)

        for img in imgs:
          img = img.resize(ORIG_RESIZE, Image.Resampling.LANCZOS) 
          old_size = img.size
          new_size = (IMAGE_SIZE_W, IMAGE_SIZE_L)
          new_im = Image.new("RGB", new_size)   ## luckily, this is already black!
          box = tuple((n - o) // 2 for n, o in zip(new_size, old_size))
          new_im.paste(img, box)
          # new_im.save(join(export_path, f"image_{i:04d}.jpg"),"PNG")
          img = new_im
          
          data.append(np.asarray(img) / 255)
          # data.append(np.asarray(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)) / 255)
          # data.append(np.asarray(img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)) / 255)
          # data.append(np.asarray(img.transpose(Image.Transpose.ROTATE_180)) / 255)

      self.data = np.array(data, dtype=np.float32)
      with open(img_data_path, 'wb') as f:
        np.save(f, self.data)

  def plot_image(self, img):
    shape = self.img_dim + (3,)
    plt.imshow(img.reshape(shape), cmap='gray', interpolation='nearest')
    plt.axis('off')

  def export_all(self):
    export_path = join(export_dir, self.datasetid)
    Path(export_path).mkdir(parents=True, exist_ok=True)

    # rnd_idx = rnd.permutation(len(data))
    fig = plt.figure(figsize=(1, 1), frameon=False, dpi=200)

    # Dump all data
    for i, img in enumerate(self.data[:50]): # [rnd_idx[:10]]
      # plt.subplot(2, 5, i + 1)
      # self.plot_image(img)
      matplotlib.image.imsave(join(export_path, f"image_{i:04d}.png"), img)
      # plt.savefig(join(export_path, f"image_{i:04d}.jpg"), bbox_inches='tight', pad_inches=0)

from tensorflow.keras.layers import (Input, Conv2D, LeakyReLU, Activation,
                                     Dropout, Flatten, Dense, Reshape,
                                     BatchNormalization, UpSampling2D,
                                     Conv2DTranspose, Lambda)
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
import numpy as np
import tensorflow.keras.backend as K
from functools import partial
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow as tf

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def get_activation(activation_fn_name):
  """Get the activation layer from the name of the function."""
  if activation_fn_name == 'leaky_relu':
    return LeakyReLU(alpha=0.2)
  return Activation(activation_fn_name)


def set_trainable(model, value):
  """Set each layer of a model as trainable."""
  model.trainable = value
  for l in model.layers:
    l.trainable = value


def get_optimizer(optimizer_name, learning_rate):
  """Get an optimizer by name."""
  if optimizer_name == 'adam':
    return Adam(learning_rate=learning_rate)
  if optimizer_name == 'rmsprop':
    return RMSprop(learning_rate=learning_rate)
  return Adam(learning_rate=learning_rate)


def wasserstein_loss(y_true, y_pred):
  """Wasserstein loss function."""
  return -K.mean(y_true * y_pred)


class WGANGP(object):
  """Implementation of a WGAN-GP using Keras."""

  def __init__(self, input_shape, critic_conv_filters, critic_conv_kernel_size,
               critic_conv_strides, critic_activation, critic_dropout_rate,
               critic_learning_rate, generator_initial_dense_layer_shape,
               generator_upsample, generator_conv_filters,
               generator_conv_kernel_size, generator_conv_strides,
               generator_batch_norm_momentum, generator_activation,
               generator_dropout_rate, generator_learning_rate, optimizer,
               gradient_weight, z_dim, batch_size):
    # Build the critic.
    critic_input = Input(shape=input_shape, name='critic_input')
    x = critic_input
    weight_init = RandomNormal(mean=0.0, stddev=0.02)
    for i in range(len(critic_conv_filters)):
      x = Conv2D(filters=critic_conv_filters[i],
                 kernel_size=critic_conv_kernel_size[i],
                 strides=critic_conv_strides[i], padding='same',
                 kernel_initializer=weight_init,
                 name='critic_conv_{}'.format(i))(x)
      x = get_activation(critic_activation)(x)
      if critic_dropout_rate:
        x = Dropout(rate=critic_dropout_rate)(x)
    x = Flatten()(x)
    critic_output = Dense(1, activation=None, kernel_initializer=weight_init)(x)
    self.critic = Model(critic_input, critic_output)

    self.z_dim = z_dim

    # Build the generator.
    generator_input = Input(shape=(z_dim,), name='generator_input')
    x = generator_input
    x = Dense(np.prod(generator_initial_dense_layer_shape),
              kernel_initializer=weight_init)(x)
    if generator_batch_norm_momentum:
      x = BatchNormalization(momentum=generator_batch_norm_momentum)(x)
    x = get_activation(generator_activation)(x)
    x = Reshape(target_shape=generator_initial_dense_layer_shape)(x)
    if generator_dropout_rate:
      x = Dropout(rate=generator_dropout_rate)(x)
    for i in range(len(generator_upsample)):
      if generator_upsample == 2:
        x = UpSampling2D()(x)
        x = Conv2D(filters=generator_conv_filters[i],
                   kernel_size=generator_conv_kernel_size[i],
                   strides=generator_conv_strides[i], padding='same',
                   kernel_initializer=weight_init,
                   name='generator_conv_{}'.format(i))(x)
      else:
        x = Conv2DTranspose(filters=generator_conv_filters[i],
                            kernel_size=generator_conv_kernel_size[i],
                            strides=generator_conv_strides[i], padding='same',
                            kernel_initializer=weight_init,
                            name='generator_conv_{}'.format(i))(x)
      x = get_activation(generator_activation)(x)
      # print(x.shape, generator_conv_filters[i], generator_conv_kernel_size[i], generator_conv_strides[i])
      if i == (len(generator_upsample) - 1):
        break
      if generator_batch_norm_momentum:
        x = BatchNormalization(momentum=generator_batch_norm_momentum)(x)
    generator_output = Activation('tanh')(x)
    self.generator = Model(generator_input, generator_output)

    # Compile the critic.
    set_trainable(self.generator, False)

    real_img = Input(shape=input_shape)
    z_disc = Input(shape=(z_dim,))
    fake_img = self.generator(z_disc)

    real = self.critic(real_img)
    fake = self.critic(fake_img)

    def interpolate_inputs(inputs):
      alpha = K.random_uniform((batch_size, 1, 1, 1))
      return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    self.interpolated_img = Lambda(interpolate_inputs)([real_img, fake_img])
    interpolated_prediction = self.critic(self.interpolated_img)

    self.critic_model = Model(inputs=[real_img, z_disc],
                              outputs=[real, fake, interpolated_prediction])

    self.critic_model.compile(
        loss=[wasserstein_loss, wasserstein_loss, self.gradient_penalty_loss],
        optimizer=get_optimizer(optimizer, critic_learning_rate),
        loss_weights=[1, 1, gradient_weight])

    set_trainable(self.generator, True)

    # Compile the generator.
    set_trainable(self.critic, False)

    model_input = Input(shape=(z_dim,))
    model_output = self.critic(self.generator(model_input))
    self.model = Model(model_input, model_output)
    self.model.compile(
        loss=wasserstein_loss,
        optimizer=get_optimizer(optimizer, generator_learning_rate))

  def gradient_penalty_loss(self, y_true, y_pred):
    """Compute the GP loss using interpolations of real and fake images."""
    gradients = K.gradients(y_pred, self.interpolated_img)[0]
    assert gradients is not None
    gradients_sqr = K.square(gradients)
    gradients_l2_norm = K.sqrt(
        K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape))))
    return K.mean(K.square(1 - gradients_l2_norm))

  def train_critic(self, X_train, batch_size):
    """Train the critic."""
    real = np.ones((batch_size, 1), dtype=np.float32)
    fake = -np.ones((batch_size, 1), dtype=np.float32)
    dummy = np.zeros((batch_size, 1), dtype=np.float32)
    rand_idx = np.random.randint(0, len(X_train), size=(batch_size,))
    X_real = X_train[rand_idx]
    noise = np.random.normal(0.0, 1.0, (batch_size, self.z_dim))
    return self.critic_model.train_on_batch([X_real, noise],
                                            [real, fake, dummy])
    
  def train_generator(self, batch_size):
    """Train the generator model."""
    real = np.ones((batch_size, 1), dtype=np.float32)
    noise = np.random.normal(0, 1, (batch_size, self.z_dim))
    return self.model.train_on_batch(noise, real)

  def train(self, X_train, batch_size, epochs, n_critic, print_every_n_epochs,
            checkpoint_path, save_every_n_epochs, initial_epoch):
    """Train the WGAN-GP model using a data generator."""
    if checkpoint_path and initial_epoch > 0:
      self.model.load_weights(
          checkpoint_path + 'model_weights_{}.hdf5'.format(initial_epoch))
    for epoch in range(initial_epoch + 1, epochs + 1):
      for _ in range(n_critic):
        d_loss = self.train_critic(X_train, batch_size)
      g_loss = self.train_generator(batch_size)
      if epoch % print_every_n_epochs == 0:
        print('Epoch: {}'.format(epoch))
        print('D Loss: {:04f}'.format(d_loss[0]))
        print('G Loss: {:04f}'.format(g_loss))
      if epoch % save_every_n_epochs == 0 and checkpoint_path:
        print('Saving after epoch: {}'.format(epoch))
        # self.model.save_weights(checkpoint_path + 'model_weights.hdf5')
        self.model.save_weights(
            checkpoint_path + 'model_weights_{}.hdf5'.format(epoch))

max_epoch = 0
for ea_checkpoint_file in checkpoint_path.iterdir():
  if ea_checkpoint_file.is_file():
    checkpoint_match = re.match(r"model_weights_([0-9]+)\.hdf5", ea_checkpoint_file.name)
    if checkpoint_match:
      max_epoch = max(max_epoch, int(checkpoint_match.group(1)))

print(f"Load epoch={max_epoch}")
oxford = ImgData('orchid')
oxford.export_all()

BATCH_SIZE = 64

wgangp = WGANGP(input_shape=(IMAGE_SIZE_W, IMAGE_SIZE_L, 3),
                critic_conv_filters=(64, 128, 256, 512),
                critic_conv_kernel_size=(5, 5, 5, 5),
                critic_conv_strides=(2, 2, 2, 2),
                critic_activation='leaky_relu',
                critic_dropout_rate=None,
                critic_learning_rate=0.0002,
                generator_initial_dense_layer_shape=(4, 4, 64),
                generator_upsample=(1, 1, 1, 1),
                generator_conv_filters=(256, 128, 64, 3),
                generator_conv_kernel_size=(5, 5, 5, 5),
                generator_conv_strides=(2, 2, 2, 2),
                generator_batch_norm_momentum=0.9,
                generator_activation='leaky_relu',
                generator_dropout_rate=None,
                generator_learning_rate=0.0002,
                optimizer='adam',
                gradient_weight=10,
                z_dim=100,
                batch_size=BATCH_SIZE)

print(wgangp.critic.summary())
print(wgangp.generator.summary())

wgangp.train(oxford.data, batch_size=BATCH_SIZE, epochs=100000, n_critic=5,
             print_every_n_epochs=10, checkpoint_path=checkpoint_path_str,
             save_every_n_epochs=1000, initial_epoch=max_epoch)

# https://github.com/matterport/Mask_RCNN/issues/2458

# wgangp.model.load_weights(
#     checkpoint_path_str + 'model_weights_{}.hdf5'.format(max_epoch))

# n_to_show = 40
# i = 0

# img = cv2.imread("export/orchid/image_0000.png")
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
# contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# cnt = contours[0]
# x,y,w,h = cv2.boundingRect(cnt)
# print(x,y,w,h)

# Path(join(generated_path_str, "orchid")).mkdir(parents=True, exist_ok=True)

# for i in range(n_to_show):
#   img_x = wgangp.generator.predict(np.random.normal(0.0, 1.0, size=(1, wgangp.z_dim)))[0]
#   img_x = np.clip(img_x, 0, 1)
#   img_x = (img_x.squeeze()*255).astype(np.uint8)
#   gen_img = Image.fromarray(img_x)
  
#   # gen_img_path = join(generated_path_str, "orchid", f"imagesrc_{i:04d}.png")
#   # gen_img.save(gen_img_path,"PNG")

#   # pil_img = cv2.imread(gen_img_path)
#   pil_img = np.array(gen_img.convert('RGB'))

#   crop = pil_img[y:y+h,x:x+w]
#   # cv2.imwrite(os.path.join("export", "orchid", f"imagers_{0:04d}.png"),crop)
#   crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

#   # pil_img = Image.open("export/orchid/image_0000.png")
#   # width, height = pil_img.size   # Get dimensions

#   # Crop the center of the image
#   blur_i = 14
#   pil_img = Image.fromarray(crop)
#   pil_img = pil_img.resize((1654, 2480), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(blur_i))
#   pil_img.save(join(generated_path_str, "orchid", f"image_{i:04d}.png"),"PNG")