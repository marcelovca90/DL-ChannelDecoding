# Getting started
Download from GitHub

    git clone git@github.com:gruberto/DL-ChannelDecoding.git
    cd DL-ChannelDecoding/docker
    
Build docker container which contains jupyter

    ./build_jupyter.sh

# Start Jupyter
Start jupyter with theano on a CPU

    ./run_jupyter_cpu.sh theano

Start jupyter with theano on a GPU

    ./run_jupyter_gpu.sh theano
    
Access Jupyter Notebooks in a browser on

    http://[ip-adress]:8888

# Local environment
    conda create -n gruberto115 python==3.6.13
    conda activate gruberto115
    conda install cudatoolkit=10.0 cudnn=7.6 -c=conda-forge 
    pip install --upgrade tensorflow-gpu==1.15.0 keras==2.3.1 matplotlib

# Notes
## Local environment (GTX 960)
    Fit started at 2021-06-14 02:04:53.869813.
    Fit finished at 2021-06-14 02:08:56.013735 (took 242.144 [s]).
## Google Colaboratory (Tesla T4)
    Fit started at 2021-06-14 05:05:07.490770.
    Fit finished at 2021-06-14 05:08:30.103033 (took 202.612 [s]).
