# Local usage

## Introduction

To run training and prediction on a local GPU or CPU, you can follow these steps:

We 

1. Clone the repository
```bash
git clone "https://github.com/emdeh/rmit-img-classifier.git"
```

2. cd into the project folder
```bash
cd img-classifier
```

3. Run the setup script to install the enviornment and the necessary dependencies, and retrieve the dataset:
```bash
source local-setup.sh # Make sure to use source
```

4. If the environment does not activate automatically, start it with:
```bash
conda activate local-env.yaml
```

5. Start the training process as described on the /home/emdeh/udacity-course/img-classifier/README.md