'''

tensorrt int8 量化,序列化engine
'''

import os
# import tensorrt as trt
# import pycuda.autoinit
# import pycuda.driver as cuda
from calibrator import Calibrator
import numpy as np
import time
from cuda import cudart
import tensorrt as trt
import ctypes


# add verbose
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) # ** engine可视化 **
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
# layernorm = ctypes.CDLL("./layerNormPlugin/layerNormKernel.so")
groupnorm = ctypes.CDLL("./groupNormPlugin/groupNormKernel.so")

# create tensorrt-engine
# fixed and dynamic
def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="",\
              int8_mode=False, calibration_stream=None, calibration_table_path="", save_engine=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(network_flags) as network,\
                builder.create_builder_config() as config, \
                trt.OnnxParser(network, TRT_LOGGER) as parser, \
                trt.Runtime(TRT_LOGGER) as runtime:
            
            # parse onnx model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing') 
                parser.parse(model.read(),path="./models/ac76aa24-2e13-11ee-bebb-0242ac110009")
                # parser.parse(onnx_file_path)

                # assert network.num_layers > 0, 'Failed to parse ONNX model. \
                #             Please check if the ONNX model is compatible '
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))        
            
            # build trt engine
            builder.max_batch_size = max_batch_size
            config.max_workspace_size =  8 * (2 ** 30)  # 8 GB # 1GB
           
            if int8_mode:
                config.set_flag(trt.BuilderFlag.INT8)
                assert calibration_stream, 'Error: a calibration_stream should be provided for int8 mode'
                config.int8_calibrator = Calibrator(calibration_stream, calibration_table_path)
                print('Int8 mode enabled')

            for layer in network:
                if "GroupNorm" in layer.name:
                    layer.precision = trt.DataType.FLOAT

            # Build engine and do int8 calibration.
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            if engine is None:
                print('Failed to create the engine')
                return None   
            print("Completed creating the engine")
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(plan)
            return engine
        
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)
