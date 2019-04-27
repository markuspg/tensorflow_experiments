#!/usr/bin/env python3

################################################################################
##
## Copyright 2019 Markus Prasser
##
## This file is part of tensorflow_experiments.
##
##  tensorflow_experiments is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  tensorflow_experiments is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with tensorflow_experiments.  If not, see <http://www.gnu.org/licenses/>.
##
################################################################################

import os
import random
import tensorflow as tf

workspaceDir = "/home/maprasser/workspace"

def CreateModel(argLabelsLength):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(8192,1)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(argLabelsLength, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def main():
    # load all WAVE files into a list and their tags into another list
    labels = list()
    labelsList = list()
    tensorsList = list()
    for root, dirs, names in tf.io.gfile.walk(workspaceDir):
        label = root.split('/')[-1]
        # skip the "workspace" directory itself
        if label == "workspace":
            continue
        labels.append(label)
        labelIdx = labels.index(label)
        for filename in names:
            labelsList.append(labelIdx)
            tensorsList.append(
              tf.audio.decode_wav(
                tf.io.read_file(os.path.join(root, filename)),
                desired_channels=1, desired_samples=8192)[0])
    assert len(labelsList) == len(tensorsList)

    # shuffle the two lists simultaneously
    shuffleList = list(zip(labelsList, tensorsList))
    random.shuffle(shuffleList)
    labelsList, tensorsList = zip(*shuffleList)

    # split into the three categories: training, testing, validation
    testData = list()
    testLabels = list()
    trainData = list()
    trainLabels = list()
    validationData = list()
    validationLabels = list()
    random.seed()
    for combinedItem in zip(labelsList, tensorsList):
        divider = random.random()
        if divider < 0.7:
            trainLabels.append(combinedItem[0])
            trainData.append(combinedItem[1])
        elif divider >= 0.7 and divider < 0.9:
            validationLabels.append(combinedItem[0])
            validationData.append(combinedItem[1])
        else:
            testLabels.append(combinedItem[0])
            testData.append(combinedItem[1])
    assert len(testData) + len(trainData) + len(validationData) \
      == len(tensorsList)
    assert len(testLabels) + len(trainLabels) + len(validationLabels) \
      == len(labelsList)

    testTensor = tf.stack(testData)
    testLabelsTensor = tf.stack(testLabels)
    trainTensor = tf.stack(trainData)
    trainLabelsTensor = tf.stack(trainLabels)

    checkpointPath = "training/cp-{epoch:04d}.ckpt"
    checkPointDir = os.path.dirname(checkpointPath)
    cpCallback = tf.keras.callbacks.ModelCheckpoint(checkpointPath,
                                                    save_weights_only=True,
                                                    verbose=1)
    trainModel = CreateModel(len(labels))
    trainModel.save_weights(checkpointPath.format(epoch=0))
    trainModel.fit(trainTensor, trainLabelsTensor, callbacks=[cpCallback],
                   epochs=5)
    tfliteModel = tf.function(lambda x : trainModel(x))
    concrete_func = tfliteModel.get_concrete_function(
      tf.TensorSpec(trainModel.inputs[0].shape, trainModel.inputs[0].dtype))
    converter = tf.lite.TFLiteConverter.from_concrete_function(concrete_func)
    tfliteModel = converter.convert()
    with open("speechModel.tflite", 'wb') as tfliteModelFile:
        tfliteModelFile.write(tfliteModel)
    trainModel.save("speechModel.h5")

    testModelA = CreateModel(len(labels))
    testModelA.load_weights(tf.train.latest_checkpoint(checkPointDir))
    loss, acc = testModelA.evaluate(testTensor, testLabelsTensor)
    print("Test model 'A' accuracy: {:5.2f}%".format(100 * acc))

    testModelB = tf.keras.models.load_model("speechModel.h5")
    loss, acc = testModelB.evaluate(testTensor, testLabelsTensor)
    print("Test model 'B' accuracy: {:5.2f}%".format(100 * acc))

if __name__ == "__main__":
    main()

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
