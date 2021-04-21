Load training and validation data as folders for under data here. Testing data ground truth is not provided, nor used in the original paper DCTNet (validation data is used instead to assess accuracy).
Folders should be of the format.
data
```
data
├── train
├── val
└── README.md
```

Download imagenet 2012 ILSVRC2012 training and validation data for tasks 1 and 2 from the following URL:
http://image-net.org/challenges/LSVRC/2012/2012-downloads

Note: training data is very large (138GB) for task 1 & 2 and should be downloaded with a download manager and the folder should symlink to the downloaded location. Go to the data folder and its README for more details on data preparation.

The 'main' folder contains the main files for running the code, including the testing loop and the resnet model. The 'utility' folder contains other helper code, including the dataloader and functions used for transforming/preprocessing the input images.

# Run the code

Clone the repository: 

```bash
git clone https://github.com/TakShimoda/CP8322-Deep-Learning-in-Computer-Vision-Course-Project.git
```

Install the dependencies(it is recommended to use a virtual environment):

```bash
cd CP8322-Deep-Learning-in-Computer-Vision-Course-Project/classification
pip install -r requirements.txt
```

To run the code to be tested on ILSVRC 2012 validation dataset, run the code by setting the batch size, whether to use DCT, and whether to load a pretrained model or not.
For example, to run on a batch size of 200 for the baseline ResNet-50 that's pretrained on ILSVRC 2012, run:

```bash
cd main
python3 resnet_eval.py --batch 200 --DCT False --pretrained True
```
