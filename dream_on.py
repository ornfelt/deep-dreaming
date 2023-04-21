'''
Some info on various layers, so you know what to expect
depending on which layer you choose:

layer 1: wavy
layer 2: lines
layer 3: boxes
layer 4: circles?
layer 6: dogs, bears, cute animals.
layer 7: faces, buildings
layer 8: fish begin to appear, frogs/reptilian eyes.
layer 10: Monkies, lizards, snakes, duck

Choosing various parameters like num_iterations, rescale,
and num_repeats really varies on which layer you're doing.


We could probably come up with some sort of formula. The
deeper the layer is, the more iterations and
repeats you will want.

Layer 3: 20 iterations, 0.5 rescale, and 8 repeats is decent start
Layer 10: 40 iterations and 25 repeats is good.
'''
from deepdreamer import model, load_image, recursive_optimize
import numpy as np
import PIL.Image
import cv2
import os
import random

layer_tensor = model.layer_tensors[3]

dream_name = 'starry_night'

x_size = 800
y_size = 450

created_count = 0
max_count = 50
for i in range(0, 9999999999999999):

    if os.path.isfile('dream/{}/img_{}.jpg'.format(dream_name, i+1)):
        print('{} already exists, continuing along...'.format(i+1))

    else:
        img_result = load_image(filename='dream/{}/img_{}.jpg'.format(dream_name, i))

        # this impacts how quick the "zoom" is
        x_trim = 2
        y_trim = 1

        img_result = img_result[0+x_trim:y_size-y_trim, 0+y_trim:x_size-x_trim]
        img_result = cv2.resize(img_result, (x_size, y_size))

        # Use these to modify the general colors and brightness of results.
        # results tend to get dimmer or brighter over time, so you want to
        # manually adjust this over time.

        # +2 is slowly dimmer
        # +3 is slowly brighter
        img_result[:, :, 0] += 2  # reds
        img_result[:, :, 1] += 2  # greens
        img_result[:, :, 2] += 2  # blues
        
        #img_result[:, :, 0] += random.choice([3, 4])  # reds
        #img_result[:, :, 1] += random.choice([3, 4])  # greens
        #img_result[:, :, 2] += random.choice([3, 4])  # blues

        img_result = np.clip(img_result, 0.0, 255.0)
        img_result = img_result.astype(np.uint8)

        img_result = recursive_optimize(layer_tensor=layer_tensor,
                                        image=img_result,
                                        num_iterations=15,
                                        step_size=1.0,
                                        rescale_factor=0.7,
                                        num_repeats=1,
                                        blend=0.2)

        img_result = np.clip(img_result, 0.0, 255.0)
        img_result = img_result.astype(np.uint8)
        result = PIL.Image.fromarray(img_result, mode='RGB')
        result.save('dream/{}/img_{}.jpg'.format(dream_name, i+1))

        created_count += 1
        if created_count > max_count:
            break