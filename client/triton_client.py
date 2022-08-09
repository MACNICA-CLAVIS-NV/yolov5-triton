#!/usr/bin/env python

# Copyright (c) 2021 MACNICA-CLAVIS-NV

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from attrdict import AttrDict

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import tritonclient.grpc.model_config_pb2 as mc


def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 1:
        raise Exception("expecting 1 output, got {}".format(
            len(model_metadata.outputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata.name + "' output type is " +
                        output_metadata.datatype)

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    # output_batch_dim = (model_config.max_batch_size > 0)
    # non_one_cnt = 0
    # for dim in output_metadata.shape:
    #     if output_batch_dim:
    #         output_batch_dim = False
    #     elif dim > 1:
    #         non_one_cnt += 1
    #         if non_one_cnt > 1:
    #             raise Exception("expecting model output to be a vector")

    # Model input must have 3 dims, either CHW or HWC (not counting
    # the batch dimension), either CHW or HWC
    input_batch_dim = (model_config.max_batch_size > 0)
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata.name,
                   len(input_metadata.shape)))

    if type(input_config.format) == str:
        FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
        input_config.format = FORMAT_ENUM_TO_INT[input_config.format]

    if ((input_config.format != mc.ModelInput.FORMAT_NCHW) and
        (input_config.format != mc.ModelInput.FORMAT_NHWC)):
        raise Exception("unexpected input format " +
                        mc.ModelInput.Format.Name(input_config.format) +
                        ", expecting " +
                        mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
                        " or " +
                        mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))

    if input_config.format == mc.ModelInput.FORMAT_NHWC:
        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
    else:
        c = input_metadata.shape[1 if input_batch_dim else 0]
        h = input_metadata.shape[2 if input_batch_dim else 1]
        w = input_metadata.shape[3 if input_batch_dim else 2]

    return (model_config.max_batch_size, input_metadata.name,
            output_metadata.name, c, h, w, input_config.format,
            input_metadata.datatype)


class TritonClientError(Exception):
    pass


class TritonClient():

    def __init__(self, url='localhost:8000'):
        self.request = None
        self.response = None

        try:
            self.client = httpclient.InferenceServerClient(
                url=url, verbose=False, concurrency=1
            )
        except Exception as e:
            print('could not create client: {}'.format(e))
            raise TritonClientError(str(e))

    def parse_model(self, model_name: str, model_version=''):
        
        if not self.client.is_model_ready(model_name):
            raise TritonClientError('model loading failure')

        try:
            model_metadata = self.client.get_model_metadata(
                model_name=model_name, model_version=model_version
            )
        except InferenceServerException as e:
            print('Could not retrive metadata: {}'.format(e))
            raise TritonClientError(str(e))

        try:
            model_config = self.client.get_model_config(
                model_name=model_name, model_version=model_version
            )
        except InferenceServerException as e:
            print('Could not retrive config: {}'.format(e))
            raise TritonClientError(str(e))

        self.model_metadata = AttrDict(model_metadata)
        self.model_config = AttrDict(model_config)
        self.model_name = model_name
        self.model_version = model_version

        self.max_batch_size, self.input_name, self.output_name, \
        self.c, self.h, self.w, self.format, self.dtype = parse_model(
            self.model_metadata, self.model_config
        )
        print('max_batch_size: {}'.format(self.max_batch_size))
        print('input_name    : {}'.format(self.input_name))
        print('output_name   : {}'.format(self.output_name))
        print('C             : {}'.format(self.c))
        print('h             : {}'.format(self.h))
        print('w             : {}'.format(self.w))
        print('format        : {}'.format(self.format))
        print('dtype         : {}'.format(self.dtype))

    def infer(self, image, class_count=0):
        if self.max_batch_size > 0:
            image = image[np.newaxis, :]

        inputs = [httpclient.InferInput(self.input_name, image.shape, self.dtype)]
        inputs[0].set_data_from_numpy(image)

        outputs = [httpclient.InferRequestedOutput(self.output_name, class_count=class_count)]

        try:
            self.request = self.client.async_infer(
                self.model_name, inputs, request_id='0',
                model_version=self.model_version, outputs=outputs
            )
        except InferenceServerException as e:
            print('Inference failed: {}'.format(e))
            raise TritonClientError(str(e))

    def get_results(self):
        if self.request is None:
            return None

        self.response = self.request.get_result()
        output_array = self.response.as_numpy(self.output_name)
        if self.max_batch_size <= 0:
            output_array = output_array[np.newaxis, :]

        return output_array
