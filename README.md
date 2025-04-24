This is a homework repository for Deep Learning spring 2025 NCKU.

The training record and the trained model are saved in the TrainedModel directory.

prepareLoader.py contains function and class related to Dataset, Dataloader and ChannelSelector to conduct the experiment of Task A.

ChannelWise_MeanSd.py contains function to compute the mean and std of each channel of a image in train data.

Models' classes are written in AlexNet.py(add BatchNorm), TaskA.py, ResNet.py, TaskB.py

Trainings are all conducted with Adam optimizer and ReduceLROnPlateau scheduler.

train_MODELNAME.py files are the training setting of models repectively. 