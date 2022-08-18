# MIT License
#
# Copyright (c) 2022 MACNICA-CLAVIS-NV
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import torch

BATCH_SIZE = 1

def main():
    model_name = 'yolov5s'
    if len(sys.argv) > 1:
        model_name = sys.argv[1]

    # Download the model file
    model = torch.hub.load('ultralytics/yolov5', model_name).to('cuda')

    # Test
    img = 'https://ultralytics.com/images/zidane.jpg'
    results = model(img)
    results.print()

    # Convert to ONNX
    onnx_file = '{}.onnx'.format(model_name)
    model = model.to('cpu')
    torch.onnx.export(
        model, torch.zeros(BATCH_SIZE, 3, 384, 640), onnx_file, 
        verbose=True, input_names=["input"], output_names=["output"],
        # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        export_params=True, opset_version=11)

if __name__ == '__main__':
    main()