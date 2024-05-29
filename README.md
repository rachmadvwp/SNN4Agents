# SNN4Agents: A Framework for Developing Energy-Efficient Spiking Neural Networks for Autonomous Agents
SNN4Agents: A Framework for Developing Energy-Efficient Spiking Neural Networks for Autonomous Agents

## Create Conda Environment (if required): 
```
conda create --name snn4agents python=3.8
```

## Installation: 
Ensure to fulfill the library requirements:
```
pip install numpy torch torchvision
```

## Preparation: 
Prepare the original dataset (n-cars_test & n-cars_train) as shown like this figure. 
<p align="left"><img width="25%" src="docs/ncars_folders.png"/></p>

Then, generate the modified dataset (N_cars) using matlab scripts and will get N_cars folder.   

Afterwards, create "Trained_100" folder and run the example below.

## Example of command to run the code:
```
CUDA_VISIBLE_DEVICES=0 python3 main.py --filenet ./net/net_1_4a32c3z2a32c3z2a_100_100_no_ceil.txt --fileresult res_100x100_noceil_wghbit_C0b32_C1b32_F1b32_F2b32 --batch_size 40 --lr 1e-3 --lr_decay_epoch 20 --lr_decay_value 0.5 --threshold 0.4 --att_window 100 100 0 0 --sample_length 10 --sample_time 1 --wghbit_c0 32 --wghbit_c1 32 --wghbit_f0 32 --wghbit_f1 32
```

