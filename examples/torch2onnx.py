import numpy as np
import time
import torch
import pdb
from collections import OrderedDict
import onnx

import sys

sys.path.append(".")
sys.path.append("./lib")
import torch.nn as nn
from torch.autograd import Variable
import onnxruntime
import timeit

import argparse
# from GFPGANReconsitution import gfpgan as GFPGAN

parser = argparse.ArgumentParser("ONNX converter")
# parser.add_argument("--src_model_path", type=str, default="rlcard\\examples\\experiments\\dmc_result\\num_buffers_200\\okey\\0_10001920.pth", help="src model path")
parser.add_argument("--src_model_path", type=str, default="experiments\\dmc_result\\num_buffers_200_copy\\okey\\0_20001856.pth", help="src model path")
parser.add_argument("--dst_model_path", type=str, default="converted.onnx", help="dst model path")
parser.add_argument("--img_size", type=int, default=None, help="img size")
args = parser.parse_args()

def get_device():
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps:0")
        print("--> Running on the GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("--> Running on the GPU")
    else:
        device = torch.device("cpu")
        print("--> Running on the CPU")

    return device   

device = get_device()
model_path = args.src_model_path
onnx_model_path = args.dst_model_path
img_size = args.img_size

model = torch.load(model_path, map_location=device, weights_only=False).net

# Create dummy inputs matching DMCNet's forward method requirements
batch_size = 1
state_shape = [437]  # Example for Okey environment
action_shape = [194]  # Number of possible actions

# Create both required inputs
dummy_obs = torch.randn(batch_size, state_shape[0], device=device)
dummy_actions = torch.randn(batch_size, action_shape[0], device=device)

# state_dict = torch.load(model_path, weights_only=False)["params_ema"]
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     # stylegan_decoderdotto_rgbsdot1dotmodulated_convdotbias
#     if "stylegan_decoder" in k:
#         k = k.replace(".", "dot")
#         new_state_dict[k] = v
#         k = k.replace("dotweight", ".weight")
#         k = k.replace("dotbias", ".bias")
#         new_state_dict[k] = v
#     else:
#         new_state_dict[k] = v

# model.load_state_dict(new_state_dict, strict=False)
model.eval()

torch.onnx.export(
    model,
    (dummy_obs, dummy_actions),  # Tuple of inputs matching forward(obs, actions),
    onnx_model_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["obs", "actions"],
    output_names=["okey_action"],
    dynamo=True
)


####
try:
    original_model = onnx.load(onnx_model_path)
    passes = ["fuse_bn_into_conv"]
    optimized_model = optimizer.optimize(original_model, passes)
    onnx.save(optimized_model, onnx_model_path)
except:
    print("skip optimize.")

