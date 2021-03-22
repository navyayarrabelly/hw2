#!/bin/python

import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle
import argparse
import sys
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
# Train MLP classifier with labels

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir_sn")
#parser.add_argument("feat_dir_mfcc")
parser.add_argument("feat_dim", type=int)
parser.add_argument("train_list_videos")
parser.add_argument("val_list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")


def readFeatfile(videos_file): 
  fread = open(videos_file, "r")
  feat_list = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(videos_file).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category

  nf=0
  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath1 = os.path.join(args.feat_dir_sn, video_id + args.feat_appendix)
    # for videos with no audio, ignored in training
    features=[]
    if os.path.exists(feat_filepath1):
      features1 = np.expand_dims(np.genfromtxt(feat_filepath1, delimiter=";", dtype="float"),axis=0)
      #features1 = features1[:,250:]
    
      #features = features[:,-200:]
      
      #features = np.concatenate((features1,features2),axis=1)
      features=np.squeeze(features1)
      print(features.shape)
      feat_list.append(features)
      #print(feat_list.shape)

      label_list.append(int(df_videos_label[video_id]))
      #features = 
    else:
        nf +=1
      
    
  print("not found",nf)
  return np.array(feat_list), np.array(label_list)

if __name__ == '__main__':

  args = parser.parse_args()
  

  # 1. read all features in one array.
  


  
  X,y =readFeatfile(args.train_list_videos)
  #X= X[:,-1*args.feat_dim:]
  #X_val, y_val = readFeatfile(args.val_list_videos)
  #X_val= X_val[:,-1*args.feat_dim:]
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42,stratify= y)

  print("number of samples: %s" % X.shape[0])
 
  print(X.shape)
  print(y.shape)
  #print(y_val)
  Hl_list =[(200,100),(200,100,50),(100,50)]
  #lr_list =[]
  alpha_list=[0.1,0.15]
  Hl_list = [(200,100)]
  activation_list=['logistic','tanh','relu']
  max_iterlist =[300,400,500,600,700]
  alpha_list=[0.1]
  activation_list=['relu']
  max_iterlist=[400]
  clf = GradientBoostingClassifier(verbose=10,n_estimators=30, learning_rate=0.1, max_depth=8, random_state=42).fit(X_train, y_train)
  clf.fit(X_train,y_train)
  acc= clf.score(X_val, y_val)
  bestacc= 100
  print(acc)
  """
  for i, hl in enumerate(Hl_list): 
      for alpha in alpha_list: 
          for activation in activation_list:
            for max_iter in max_iterlist:
              #mlp = MLPClassifier(hidden_layer_sizes= hl,random_state=42,max_iter=max_iter, activation=activation, solver="sgd",alpha=alpha)   
              #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=8, random_state=42).fit(X_train, y_train)
              #mlp = SVC(cache_size=2000, decision_function_shape='ovr', kernel="poly")
              #pca = PCA(n_components=200)
              #clf = Pipeline(steps=[('pca', pca), ('mlp', mlp)])
              #clf = Pipeline(steps=[('mlp', mlp)])
              #clf.fit(X_train,y_train)
              #bestacc=0
              #bestmodel =None
              #acc= clf.score(X_val, y_val)
              #print("hl:",hl," aplha:",alpha, " activation:",activation, "max_iter",max_iter,"val acc:", acc)
              #pickle.dump(clf, open(args.output_file+str(i)+str(alpha)+str(activation)+str(max_iter), 'wb'))
              #pickle.dump(clf, open(args.output_file,'wb'))
              if(acc>bestacc): 
                  bestmodel = deepcopy(clf)
  """

  # save trained MLP in output_file
  pickle.dump(clf, open(args.output_file, 'wb'))
  print('MLP classifier trained successfully')
