Please load the ILSVRC2012 data here. It should preferably be through a symlink to the directory.

The image directory should be of the form:

```
val
├── n01440764
    ├── ILSVRC2012_val_00000293.JPEG
    ├── ILSVRC2012_val_00002138.JPEG
    .
    .
    .
    └── ILSVRC2012_val_00048969.JPEG
├── n01443537
.
.
.
└── n15075141
```

Where there are 1000 folders representing the 1000 classes for ImageNet. For the validation data, there should be 50 images in each of the folders, totalling 50,000 images.
