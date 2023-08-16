echo "preprocess"

#plugin
cd ./groupNormPlugin
make clean
make
cd ..

# cd layerNormPlugin
# make clean
# make
# cd ..

# cd splitGeLUPlugin
# make clean
# make
# cd ..

# cd seqLen2SpatialPlugin
# make clean
# make
# cd ..

export CUDA_MODULE_LOADING=LAZY

python3 01_export_pth.py
python3 02_export_onnx.py
python3 03_export_combine.py
python3 04_get_post_onnx.py
python3 04_add_LN_GN2Combine.py
python3 05_clip_16.py
#python3 05_add_fmhaplugin.py
#CUDA_MODULE_LOADING=LAZY python3 06_replace_trtexec.py


echo "get onnx success!!"

#combine
trtexec --onnx=./models/combine_0.onnx --saveEngine=combine.plan --verbose --workspace=3000 --fp16  --plugins=./groupNormPlugin/groupNormKernel.so --useCudaGraph 

#vae
trtexec --onnx=./models/vae_decoder_0.onnx --saveEngine=vae_decoder.plan --verbose  --fp16 --plugins=./groupNormPlugin/groupNormKernel.so --useCudaGraph

#clip dynamic shape
trtexec --onnx=./models/clip_encoder_1.onnx --saveEngine=clip_encoder.plan --fp16 --workspace=3000 --minShapes=input_ids:2x77 --optShapes=input_ids:2x77 --maxShapes=input_ids:2x77 --verbose --useCudaGraph

# DDIM post pro
trtexec --onnx=./models/postnet.onnx --saveEngine=postnet.plan --verbose --fp16 --useCudaGraph

echo "get trt plan success!!"



