#!/usr/bin/env python3

import os
import tensorflow as tf

workspaceDir = "/home/maprasser/workspace"

def main():
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
                desired_channels=1, desired_samples=8192))
    assert len(labelsList) == len(tensorsList)

if __name__ == "__main__":
    main()

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
