# Setting Up Lambda Labs with SSH Port Forwarding for Jupyter Notebooks

This guide explains how to set up a remote GPU instance on Lambda Labs and use SSH port forwarding to access Jupyter Notebooks from your local machine. This setup allows you to work locally in VSCode while leveraging the computational power of a remote GPU.

## 1. Provision a GPU Instance on Lambda Labs

- Sign in to your Lambda Labs account (or other remote instance used for training)
- Provision a new GPU instance with the desired specifications.
- Take note of the public IP address of the instance and your SSH login credentials.

## 2. Set Up the Remote Environment

### SSH into the Remote Instance

Use the following command to SSH into your remote instance:

```bash
ssh -i ~/.ssh/ssh_key ubuntu@remote-server-ip
```

### Install Necessary Packages
If you're using Conda, first install Conda (if not already installed), and then create a new environment:

>This can be done using the `setup.sh` script. See the README for more info, otherwise follow steps below.


### Install Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Create and activate a new Conda environment
```bash
conda create -n myenv python=3.9
conda activate myenv
```

### Install Jupyter and other dependencies
```bash
conda install jupyter notebook
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

### Install Jupyter Notebook
```bash
pip install jupyter notebook
```

### Install additional dependencies
```bash
pip install torch torchvision
```

## 3. Copy over training material. In this case:
```bash
/flowers/*
/cat_to_name.json
```

## 4. Start the Jupyter Notebook Server
Start the Jupyter Notebook server on the remote machine:

```bash
jupyter notebook --no-browser --port=8888
```

>*Note the token provided in the terminal output, which will be used to authenticate the connection.*

## 5. Set Up SSH Port Forwarding

On your local machine, set up SSH port forwarding to connect to the remote Jupyter Notebook server:

```bash
ssh -L 8888:localhost:8888 user@remote-server-ip
```

This command forwards port 8888 on your local machine to port 8888 on the remote server.

## 6. Connect to the Remote Jupyter Notebook from VSCode

### 6.1 Open VSCode
Open VSCode on your local machine.

Install the Python and Jupyter extensions if not already installed.

### 6.2. Specify Jupyter Server for Connections

Open the Command Palette in VSCode (Ctrl + Shift + P).

Search for `Python: Specify Jupyter Server for Connections`.

Enter the following URL:
`http://localhost:8888/?token=your-token`

>*Replace your-token with the actual token provided when you started the Jupyter Notebook server on the remote machine.*

## 7. Work Locally, Compute Remotely

Now, when you run a cell in your Jupyter Notebook within VSCode, the code execution occurs on the remote GPU server. You can develop and debug locally while taking advantage of the remote server's computational power.

*(super handy if you waste two days trying to get your AMD GPU working through a Linux subsystem :/ )*

## 8. Verify GPU Access
To ensure that your Jupyter Notebook is utilizing the GPU, run the following test in a notebook cell:

```bash
import torch
print(torch.cuda.is_available())
```
If `True` is printed, the GPU is accessible and ready for use.

## 9. Verify GPU connsumption

If you train on a remote GPU and if you want to monitor it, use the following command:
```bash
watch -n 1 nvidia-smi
```