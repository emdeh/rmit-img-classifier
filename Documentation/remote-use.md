# Remote usage

## Introduction

To run training and prediction on a remote GPU, you can follow these steps:

1. Retrieve the setup.sh file on the remote GPU.
```bash
wget <insert-PATH>
```

2. Make the script executable
```bash
chmod +x setup.sh
```

3. Run the setup script to install the enviornment and the necessary dependencies, and retrieve the repo and dataset:
```bash
source setup.sh # Make sure to use source
```

4. If the environment does not activate automatically, start it with:
```bash
conda activate img-classifier
```

5. Start the training process - see <INSERT-HOW-TO> for usage.

6. You may wish to monitor the remote gpu with:
```bash
watch -n 1 nvidia-smi
```