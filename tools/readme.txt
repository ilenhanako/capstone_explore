preprocessing.ipynb
Input: mergedDataset
Outputs: preprocessedDataset, seededdatsaet
Notes: initially a full pipeline, and was used in google colab, please check if you want to use this. commented out some code at cell 6,
might want to uncomment but expect an error. nonetheless should output the preprocessedDataset and the seeded dataset


resume_train.py
Input: seeededdataset or preprocessedDataset
output: a full training session (but rather than using gpu vram or using ram, it will use disk - since my disk is SSD, its speed is not too bad)
