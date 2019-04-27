#!/usr/bin/env python3

import os

workspaceDir = "/home/maprasser/workspace"

def main():
    labels = list()
    labelsList = list()
    tensorsList = list()
    for root, dirs, names in os.walk(workspaceDir):
        label = root.split('/')[-1]
        # skip the "workspace" directory itself
        if label == "workspace":
            continue
        labels.append(label)
        labelIdx = labels.index(label)
        for filename in names:
            labelsList.append(labelIdx)

if __name__ == "__main__":
    main()

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
