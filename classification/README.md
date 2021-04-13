Load training and validation data as folders for under data here. Testing data ground truth is not provided, nor used in the original paper DCTNet (validation data is used instead to assess accuracy).
Folders should be of the format.
data
├── train
├── val
├── README.md

Download imagenet 2012 ILSVRC2012 training and validation data for tasks 1 and 2 from the following URL:
http://image-net.org/challenges/LSVRC/2012/2012-downloads

Note: training data is very large (138GB) for task 1 & 2 and should be downloaded with a download manager and the folder should symlink to the downloaded location.
