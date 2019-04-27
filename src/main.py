#!/usr/bin/env python3

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

    testModel = CreateModel(len(labels))
    testModel.load_weights(tf.train.latest_checkpoint(checkPointDir))
    loss, acc = testModel.evaluate(testTensor, testLabelsTensor)
    print("Test model accuracy: {:5.2f}%".format(100 * acc))

if __name__ == "__main__":
    main()

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
