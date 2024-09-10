#!/usr/bin/env python
# coding: utf-8

class YoloKeyPoint:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.confidence = 0.0

class YoloResult:

    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.class_id = 0
        self.confidence = 0.0
        self.keypoints = []

class YoloInferenceResults:

    def __init__(self):
        self.frame = None
        self.results = []