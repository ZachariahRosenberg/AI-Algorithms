# AI Algorithms

- [Image Detection](#1.-Image-Detection)
- [Text Generation with Recurrent Neural Network](#2.-Text-Generation-with-Recurrent-Neural-Network)
- [Generating Fake People Photos with a Generative Adversarial Network](#3.-Generating-Fake-People-Photos-with-a-Generative-Adversarial-Network)
- [Reinforcement Learning to Solve Interactive Problems](#4.-Reinforcement-Learning-to-Solve-Interactive-Problems)


## 1. Image Detection
----

<br />

### 1.a Facial recognition with Haar Cascades
----

This technique uses Haar cascades, which is a simple yet powerful algorithm that looks for predetermined color contrasts in a photo to detect likely faces. [Here is a tutorial on the technique](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html)

![](https://opencv-python-tutroals.readthedocs.io/en/latest/_images/haar.png)

<br />

**To run:**

`python ./CNN/haar_cascades_face_detection.py 'path/to/image.png'`

Make sure your argument is a string path to a valid image (png, jpg, jpeg or gif). 

The function will open up your photo with bounding squares around likely faces:
![Example of Face Detection](face_detection_example.png)

<br />

### 1.b Image Classification with a Convolutional Neural Network
----

We train a CNN on the Cifar10 dataset, which is a bunch of 32x32 images of one of the following categories: 

1. airplane
2. automobile
3. bird
4. cat
5. deer
6. dog
7. frog
8. horse
9. ship
10. truck

Instead of training the entire network, we use a pre-trained base model, the [ResNet50](https://arxiv.org/abs/1512.03385). We then remove the head and add a few trainable layers on top of it. This allows us to leverage the ResNet initial layers, which are strong at detecting base outlines, hierarchies and shapes and only focus training at the end where we classify the specific categories (in this example, the CIFAR10).

![ResNet50](https://i.stack.imgur.com/gI4zT.png)

<br />

**To run:**

`python ./CNN/cnn.py`

The code will look for a network weights at path `./CNN/weights.hdf5`. If found, will skip training. Otherwise, the code will train on the CIFAR10 dataset (~several hours) and save the weights to the above path. 

It will then pick 10 random photos from the case and print to the console its prediction and level of confidence. 

Finally, it will ask you to input a string path to an image and make a prediction. 

<br />

## 2. Text Generation with Recurrent Neural Network
----

<br />

## 3. Generating Fake People Photos with a Generative Adversarial Network
----

<br />

## 4. Reinforcement Learning to Solve Interactive Problems
----

- Dynamic Programming
- DQN
- VPG
- DDPG
- PPO