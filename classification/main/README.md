To run the code to be tested on ILSVRC 2012 validation dataset, run the code by setting the batch size, whether to use DCT, and whether to load a pretrained model or not.
For example, to run on a batch size of 200 for the baseline ResNet-50 that's pretrained on ILSVRC 2012, run:

```bash
python3 resnet_eval.py --batch 200 --DCT False --pretrained True
```

