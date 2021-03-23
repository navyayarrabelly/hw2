# hw2


1.SURF feature extrcation 

2.Randomly slect 20 percent of the features from the extracted frames for each image.

python select_frames.py labels/trainval.csv surf_feat/ 0.20 selected.surf.20.csv

3. Train Kmeans model

python train_kmeans.py selected.surf.csv 150 surf.10.n.kmeans.150.model

4. Get VLAD feature vectorss

python get_vladfeat.py surf.10.n.kmeans.150.model surf_feat 150 videos.name.lst SURFvideo_vladrep


####################

1.CNN feature extraction : 
sudo python cnn_feat_extraction.py videos resnet-18/

2.Training an MLP 
python2 train_mlp.py ./resnet-18/ 512 labels/train_split.csv labels/val_split.csv models/resnet.mlp.model

3.Getting labels for Test set
python2  test_mlp.py resnet-18/ 512 labels/test_for_student.label models/resnet.mlp.model resnet_meanp.testresults.csv
