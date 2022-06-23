"""
============================
# -*- coding: utf-8 -*-
# @Time    : 2022/6/18 16:39
# @Author  : Yingjie Bai
# @FileName: onnx2TRT.py
===========================
"""
import torch
import numpy as np
import os
import onnxruntime
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
G_LOGGER = trt.Logger(trt.Logger.VERBOSE)

class MyAlgorithmSelector(trt.IAlgorithmSelector):

    def __init__(self, keepAll=True):
        super(MyAlgorithmSelector, self).__init__()
        self.keepAll = keepAll

    def select_algorithms(self, layerAlgorithmContext, layerAlgorithmList):
        print(layerAlgorithmContext.name, len(layerAlgorithmList))
        result = list((range(len(layerAlgorithmList))))
        return result

    def report_algorithms(self, modelAlgorithmContext, modelAlgorithmList):

        for i in range(len(modelAlgorithmContext)):
            context = modelAlgorithmContext[i]
            algorithm = modelAlgorithmList[i]

            print("Layer%4d:%s" % (i, context.name))
            nInput = context.num_inputs
            nOutput = context.num_outputs
            for j in range(nInput):
                ioInfo = algorithm.get_algorithm_io_info(j)
                print("    Input [%2d]:%s,%s,%s,%s" % (
                j, context.get_shape(j), ioInfo.dtype, ioInfo.strides, ioInfo.tensor_format))
            for j in range(nOutput):
                ioInfo = algorithm.get_algorithm_io_info(j + nInput)
                print("    Output[%2d]:%s,%s,%s,%s" % (
                j, context.get_shape(j + nInput), ioInfo.dtype, ioInfo.strides, ioInfo.tensor_format))
            print("    algorithm:[implementation:%d,tactic:%d,timing:%fms,workspace:%dMB]" % \
                  (algorithm.algorithm_variant.implementation,
                   algorithm.algorithm_variant.tactic,
                   algorithm.timing_msec,
                   algorithm.workspace_size))


def infer_algorithm_selector_onnx():
    builder = trt.Builder(G_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.max_workspace_size = 3 << 30
    parser = trt.OnnxParser(network, G_LOGGER)
    with open('./test.onnx', 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
    print("Succeeded parsing onnx file!")

    print(network.num_layers)

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.name == 'MatMul_14':
            layer.precision = trt.float32
            layer.get_output(0).dtype = trt.float32

    input_ids = network.get_input(0)
    bbox = network.get_input(1)
    images = network.get_input(2)
    attention_mask = network.get_input(3)


    profile.set_shape(input_ids.name, [6,709,768], [6,709,768], [6,709,768])
    profile.set_shape(bbox.name, [6,1,1,709], [6,1,1,709], [6,1,1,709])
    profile.set_shape(images.name, [6,12,709,709], [6,12,709,709], [6,12,709,709])
    profile.set_shape(attention_mask.name, [6,12,709,709], [6,12,709,709], [6,12,709,709])
    config.add_optimization_profile(profile)
    config.set_flag(trt.BuilderFlag.DEBUG)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.flags = config.flags | (1 << int(trt.BuilderFlag.STRICT_TYPES)) | (1 << int(trt.BuilderFlag.FP16))
    config.algorithm_selector = MyAlgorithmSelector(True)  # set algorithm_selector
    engineString = builder.build_serialized_network(network, config)
    with open('./test.plan', 'wb') as f:
        f.write(engineString)

infer_algorithm_selector_onnx()
