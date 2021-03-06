# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

import sys
sys.path.append("../common")

import math
import unittest
import numpy as np
import test_util as tu
import tritongrpcclient as grpcclient
import tritonhttpclient as httpclient
from tritonclientutils import np_to_triton_dtype, InferenceServerException


class LargePayLoadTest(tu.TestResultCollector):

    def setUp(self):
        self.data_type_ = np.float32
        # n GB divided by element size as tensor shape
        tensor_shape = (math.trunc(6 * (1024 * 1024 * 1024) /
                                   np.dtype(self.data_type_).itemsize),)
        self.in0_ = np.random.random(tensor_shape).astype(self.data_type_)

        small_tensor_shape = (1,)
        self.sin0_ = np.random.random(small_tensor_shape).astype(
            self.data_type_)

        self.clients_ = ((httpclient,
                          httpclient.InferenceServerClient('localhost:8000')),
                         (grpcclient,
                          grpcclient.InferenceServerClient('localhost:8001')))

    def _test_helper(self,
                     client,
                     model_name,
                     input_name='INPUT0',
                     output_name='OUTPUT0'):
        try:
            inputs = [
                client[0].InferInput(input_name, self.in0_.shape,
                                     np_to_triton_dtype(self.data_type_))
            ]
            inputs[0].set_data_from_numpy(self.in0_)
            results = client[1].infer(model_name, inputs)
            # if the inference is completed, examine results to ensure that
            # the framework and protocol do support large payload
            self.assertTrue(
                np.array_equal(self.in0_, results.as_numpy(output_name)),
                "output is different from input")

        except InferenceServerException as ex:
            # if the inference failed, inference server should return error
            # gracefully. In addition to this, send a small payload to
            # verify if the server is still functional
            inputs = [
                client[0].InferInput(input_name, self.sin0_.shape,
                                     np_to_triton_dtype(self.data_type_))
            ]
            inputs[0].set_data_from_numpy(self.sin0_)
            results = client[1].infer(model_name, inputs)
            self.assertTrue(
                np.array_equal(self.sin0_, results.as_numpy(output_name)),
                "output is different from input")

    def test_graphdef(self):
        # graphdef_nobatch_zero_1_float32 is identity model with input shape [-1]
        for client in self.clients_:
            model_name = tu.get_zero_model_name("graphdef_nobatch", 1,
                                                self.data_type_)
            self._test_helper(client, model_name)

    def test_savedmodel(self):
        # savedmodel_nobatch_zero_1_float32 is identity model with input shape [-1]
        for client in self.clients_:
            model_name = tu.get_zero_model_name("savedmodel_nobatch", 1,
                                                self.data_type_)
            self._test_helper(client, model_name)

    def test_netdef(self):
        # netdef_nobatch_zero_1_float32 is identity model with input shape [-1]
        for client in self.clients_:
            model_name = tu.get_zero_model_name("netdef_nobatch", 1,
                                                self.data_type_)
            self._test_helper(client, model_name)

    def test_onnx(self):
        # onnx_nobatch_zero_1_float32 is identity model with input shape [-1]
        for client in self.clients_:
            model_name = tu.get_zero_model_name("onnx_nobatch", 1,
                                                self.data_type_)
            self._test_helper(client, model_name)

    def test_plan(self):
        # plan_nobatch_zero_1_float32 is identity model with input shape [-1]
        for client in self.clients_:
            model_name = tu.get_zero_model_name("plan_nobatch", 1,
                                                self.data_type_)
            self._test_helper(client, model_name)

    def test_libtorch(self):
        # libtorch_nobatch_zero_1_float32 is identity model with input shape [-1]
        for client in self.clients_:
            model_name = tu.get_zero_model_name("libtorch_nobatch", 1,
                                                self.data_type_)
            self._test_helper(client, model_name, 'INPUT__0', 'OUTPUT__0')

    def test_custom(self):
        # custom_zero_1_float32 is identity model with input shape [-1]
        for client in self.clients_:
            model_name = tu.get_zero_model_name("custom", 1, self.data_type_)
            self._test_helper(client, model_name)


if __name__ == '__main__':
    unittest.main()
