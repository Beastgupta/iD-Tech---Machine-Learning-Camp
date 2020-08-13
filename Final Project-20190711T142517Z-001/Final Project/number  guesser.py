from PIL import Image, ImageFilter
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import time
import cv2
from resizeimage import resizeimage


# Loading Training Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_height = 28
image_width = 28

steps = 50000
batch_size = 32
# Taking Picture from Camera
camera_port = 0
camera = cv2.VideoCapture(camera_port)
time.sleep(0.1)
return_value, image = camera.read()
cv2.imwrite("opencv.png", image)
del(camera)
# camera_img = Image.open("opencv.png")
# camera_img = camera_img.convert('L') # Convert to Grey Color
# camera_img = camera_img.point(lambda x: 0 if x<128 else 255, '1')
# camera_img.save("result_bw.png")
# camera_img = camera_img.resize((28, 28), Image.ANTIALIAS)
# plt.imshow(camera_img)
# plt.show()

def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 for x in tv]
    return tva

x =[imageprepare('guaranteed_8.png')]#file path here

#Now we convert 784 sized 1d array to 24x24 sized 2d array so that we can visualize it
newArr=[[0 for d in range(28)] for y in range(28)]
k = 0
for i in range(28):
    for j in range(28):
        newArr[i][j]=int(x[0][k])
        k=k+1

x_train = x_train.reshape(-1, image_height, image_width, 1)

# Formatting Data
camera_img = np.array(newArr)
plt.imshow(camera_img)
plt.show()#Show / plot that image
camera_img = camera_img.reshape(-1,image_height, image_width,1)
print(type(camera_img))
print("Camera_image: ",camera_img.shape)
print(camera_img)


tf.reset_default_graph()

#Sample test
# test_img = x_test[9]
# print(type(test_img))
# print(test_img.shape)
# plt.imshow(test_img)
# plt.show()
# print(test_img)
# # plt.imshow(test_img)
# # plt.show()
# test_img = test_img.reshape(-1, image_height, image_width, 1)


# Building Model
class CNN:
    def __init__(self, image_height, image_width, channels, num_classes):
        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, channels], name="inputs")
        conv_layer_1 = tf.layers.conv2d(self.input_layer, filters=32, kernel_size=[2, 2], padding="same", activation=tf.nn.relu)
        pooling_layer_1 = tf.layers.max_pooling2d(conv_layer_1, pool_size=[2, 2], strides=2)
        flattened_pooling = tf.layers.flatten(pooling_layer_1)
        dense_layer = tf.layers.dense(flattened_pooling, 1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(dense_layer, rate=0.4, training=True)
        outputs = tf.layers.dense(dropout, num_classes)

        self.choice = tf.argmax(outputs, axis=1)
        self.probability = tf.nn.softmax(outputs)
        self.labels = tf.placeholder(dtype=tf.float32, name="labels")
        self.accuracy, self.accuracy_op = tf.metrics.accuracy(self.labels, self.choice)
        one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, dtype=tf.int32), depth=num_classes)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=outputs)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
        self.train_operation = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

cnn = CNN(28, 28, 1, 10)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    step = 0
    while step < steps:
        sess.run((cnn.train_operation, cnn.accuracy_op), feed_dict = {cnn.input_layer:x_train[step:step+batch_size], cnn.labels:y_train[step:step+batch_size]})
        step += batch_size






























































































































































































































print("[8]")
