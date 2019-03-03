# Reliable and Interpretable AI 2018 Project
Validate robustness for several neural networks with given input noise.

# Requirements
 - [VirtualBox image](https://files.sri.inf.ethz.ch/website/teaching/riai2018/materials/project/riai.ova)
 - Add `source ~/analyzer/setup_gurobi.sh` to `~/.bashrc`

# Evaluating a network
`python3 analyzer.py ../mnist_nets/mnist_relu_3_10.txt ../mnist_images/img0.txt 0.01`
