# LAMP-FPGA

This is an FPGA implementation of [Learned Approximate Matrix Profile (LAMP)](https://github.com/zpzim/LAMP-ICDM2019) algorithm on Ultra96-V2 board. LAMP-FPGA takes a time series as input and predicts its matrix profile values for a particular window size. You can read more at the [Matrix Profile Homepage](http://www.cs.ucr.edu/~eamonn/MatrixProfile.html).

# Folder Structure
    .
    ├── script                  # Scripts for generating the compiled model for use on FPGA and evalution
    ├── files                   # Input dataset sample, calibration data, pre-trained LAMP model
    ├── src                     # Source files for executing the model on FPGA
    ├── LICENSE
    └── README.md
# Build Instructions

In this section we show how to deploy LAMP on Xilinx Deep Processing Unit (DPU) and implement it on the FPGA given a pre-trained model on CPU. The instructions require [Vitis-AI](https://github.com/Xilinx/Vitis-AI) installed on the system.
### 1. Freezing Tensorflow graph 
  The Vitis-AI flow requires  a frozen model for quantization and optimization steps. A frozen model contains information about the graph and checkpoint variables, saving these hyperparameters as constants within the graph structure. This allows fusing some of the layers together for deployment on DPU. We can generate a binary protobuf (.pb) file by running the <code>freeze_graph.py</code>  script
  
  ```shell
  python freeze_graph.py input_model
  ```
  where <code>input_model</code> is the pre-trained LAMP model.
  
  ### 2. Quantization
  
  We will quantize the weights/biases and activations of the model to improve the performance of the model inference on FPGA. Currently, Xilinx DPU only supports 8 bit models, so we quantize everything to 8 bits.
```shell
vai_q_tensorflow quantize 
                 --input_frozen_graph frozen_graph.pb 
                 --input_fn input_func.calib_input
                 --output_dir quantized 
                 --input_nodes input_1 
                 --output_nodes reshape_1/Reshape 
                 --input_shapes ?,256,1,32 
                 --calib_iter 32
```
<code>frozen_graph.pb</code> is the frozen model generated in the previous step, <code>input_func</code> is the python file that generates the input data for quantizer (since there is no backpropagation step here, the unlabeled dataset is sufficient), and <code>calib_iter</code> is the number of iterations for calibrating the activations, we noticed that values larger than 32 do not increase the quantizer accuracy by a large degree.
### 3. Evaluation
We will test the accuracy of the generated quantized model before deploying it to the FPGA. 

```shell
python evaluate.py
```
<code>evaluate.py</code> reads in the Tensorflow frozen binary graph, runs the inference and reports the mean squared error and mean absolute percentage error by comparing the model output with the labels (matrix profile values). 
### 4. Compilation
 Vitis-AI Docker image does not support Ultra96-v2 board, we need to generate the DPU configuration file (Ultra96.dcf) required in the compile step by first downloading the DPU Hardware Handoff file [dpu.hwh](https://www.xilinx.com/bin/public/openDownload?filename=pynqdpu.dpu.ultra96.hwh) and then running the following command
 
```shell
dlet -f dpu.hwh
```
dlet is a host tool that extracts the DPU information from the input file and generates the configuration file.
 
 Next, we will compile the model for the target hardware
 ```shell
vai_c_tensorflow --frozen_pb quantized\deploy_model.pb 
                  --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ultra96/arch.json 
                  --output_dir . 
                  --net_name lamp
 ```
<code>arch.json</code> is located in the files directory. Since, Sigmoid and Global Average Pool layers are not supported by DPU, the command generates four kernels, two of which will be deployed to the FPGA and the other two will be implemented on the host CPU. Xilinx DNNK API has an issue with loading the static libraries that start with the same name, in order to fix this issue, we run the same command one more time setting the output model name to <code>dense_lamp</code>, later, we will use <code>lamp_0.elf</code> and <code>dense_lamp_2.elf</code> when loading the kernels.

### 5. Running inference
First we need to install dpu-pynq on the Ultra96 board. Open a terminal and run 
```shell
git clone --recursive --shallow-submodules https://github.com/Xilinx/DPU-PYNQ
cd DPU-PYNQ/upgrade
make

pip3 install pynq-dpu
```
The build process might take up to an hour. After that, we can run the models on board by executing <code>lamp_dpu.ipynb</code>. The notebook takes a time series dataset as input and writes the predictions in <code>predict.txt</code> file.

