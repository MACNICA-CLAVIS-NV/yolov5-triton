#!/usr/bin/env python

# MIT License
#
# Copyright (c) 2021 MACNICA-CLAVIS-NV
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import time
import numpy as np


class IntervalCounter():
    '''A counter to measure the interval between the measure method calls.

    Attributes:
        numSamples: Number of samples to calculate the average.
        samples: Buffer to store the last N intervals.
        lastTime: Last time stamp
        count: Total counts
    '''

    def __init__(self, numSamples):
        '''
        Args:
            numSamples(int): Number of samples to calculate the average.
        '''
        self.numSamples = numSamples
        self.samples = np.zeros(self.numSamples)
        self.lastTime = time.time()
        self.count = 0

    def __del__(self):
        pass

    def measure(self):
        '''Measure the interval from the last call.

        Returns:
            The interval time count in second.
            If the number timestamps captured in less than numSamples,
            None will be returned.
        '''
        curTime = time.time()
        elapsedTime = curTime - self.lastTime
        self.lastTime = curTime
        self.samples = np.append(self.samples, elapsedTime)
        self.samples = np.delete(self.samples, 0)
        self.count += 1
        if self.count > self.numSamples:
            return np.average(self.samples)
        else:
            return None
