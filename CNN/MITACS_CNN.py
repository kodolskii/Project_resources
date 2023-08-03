import tensorflow.keras as keras
from keras import layers,models
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,Add, AveragePooling2D, Flatten, Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt

Na = 4 #Number of atoms
vfecol = 56

y = open('NN4_Co_Vfe.txt')
Covfe = []
for row in y:
  Covfe.append([float(x) for x in row.split()])
print(Covfe[0])

y = open('NN4_Cr_Vfe.txt')
Crvfe = []
for row in y:
  Crvfe.append([float(x) for x in row.split()])

y = open('NN4_Fe_Vfe.txt')
Fevfe = []
for row in y:
  Fevfe.append([float(x) for x in row.split()])

y = open('NN4_Ni_Vfe.txt')
Nivfe = []
for row in y:
  Nivfe.append([float(x) for x in row.split()])

print(Covfe[0])

hea_vac_Co = []
for i in range(0,len(Covfe)):
  hea_vac_Co.append((Covfe[i][:55]))
hea_vac_Cr = []
for i in range(0,len(Crvfe)):
  hea_vac_Cr.append(Crvfe[i][:55])
hea_vac_Fe = []
for i in range(0,len(Fevfe)):
  hea_vac_Fe.append(Fevfe[i][:55])
hea_vac_Ni = []
for i in range(0,len(Nivfe)):
  hea_vac_Ni.append(Nivfe[i][:55])
for i in range(0,len(hea_vac_Co)):
  for j in range(0,len(hea_vac_Co[i])):
    hea_vac_Co[i][j] = (hea_vac_Co[i][j])
for i in range(0,len(hea_vac_Cr)):
  for j in range(0,len(hea_vac_Cr[i])):
    hea_vac_Cr[i][j] = (hea_vac_Cr[i][j])
for i in range(0,len(hea_vac_Fe)):
  for j in range(0,len(hea_vac_Fe[i])):
    hea_vac_Fe[i][j] = (hea_vac_Fe[i][j])
for i in range(0,len(hea_vac_Ni)):
  for j in range(0,len(hea_vac_Ni[i])):
    hea_vac_Ni[i][j] = (hea_vac_Ni[i][j])

print((hea_vac_Co[0][54]))
print(len(hea_vac_Ni))

print(len(hea_vac_Co[0]))
type(hea_vac_Co[0][0])

datasetCo = []
datasetCr = []
datasetFe = []
datasetNi = []

datasetCo = []
for i in range(len(hea_vac_Co)):
  newimg = np.zeros((24,24),dtype = float)
  def minimat(inicordx,inicordy,finalcordx,finalcordy,heaval):
    for i in range(inicordx-1,finalcordx):
      for j in range(inicordy-1,finalcordy):
        newimg[i][j] = heaval
    return newimg
  minimat(11,11,14,14,hea_vac_Co[0][54])
  minimat(9,9,10,10,hea_vac_Co[i][0])
  minimat(9,11,10,12,hea_vac_Co[i][1])
  minimat(9,13,10,14,hea_vac_Co[i][2])
  minimat(9,15,10,16,hea_vac_Co[i][3])
  minimat(11,15,12,16,hea_vac_Co[i][4])
  minimat(13,15,14,16,hea_vac_Co[i][5])
  minimat(15,15,16,16,hea_vac_Co[i][6])  
  minimat(15,13,16,14,hea_vac_Co[i][7])
  minimat(15,11,16,12,hea_vac_Co[i][8])
  minimat(15,9,16,10,hea_vac_Co[i][9])
  minimat(13,9,14,10,hea_vac_Co[i][10])
  minimat(11,9,12,10,hea_vac_Co[i][11])
  minimat(7,11,8,14,hea_vac_Co[i][12])
  minimat(11,17,14,18,hea_vac_Co[i][13])
  minimat(17,11,18,14,hea_vac_Co[i][14])
  minimat(11,7,14,8,hea_vac_Co[i][15])

  minimat(7,7,8,8,hea_vac_Co[i][16])
  minimat(17,17,18,18,hea_vac_Co[i][16])

  minimat(7,17,8,18,hea_vac_Co[i][17])
  minimat(17,7,18,8,hea_vac_Co[i][17])

  minimat(3,7,6,8,hea_vac_Co[i][18])
  minimat(3,9,6,10,hea_vac_Co[i][19])
  minimat(3,11,6,12,hea_vac_Co[i][20])
  minimat(3,13,6,14,hea_vac_Co[i][21])
  minimat(3,15,6,16,hea_vac_Co[i][22])
  minimat(3,17,6,18,hea_vac_Co[i][23])

  minimat(7,19,8,22,hea_vac_Co[i][24])
  minimat(9,19,10,22,hea_vac_Co[i][25])
  minimat(11,19,12,22,hea_vac_Co[i][26])
  minimat(13,19,14,22,hea_vac_Co[i][27])
  minimat(15,19,16,22,hea_vac_Co[i][28])
  minimat(17,19,18,22,hea_vac_Co[i][29])

  minimat(19,17,22,18,hea_vac_Co[i][30])
  minimat(19,15,22,16,hea_vac_Co[i][31])
  minimat(19,13,22,14,hea_vac_Co[i][32])
  minimat(19,11,22,12,hea_vac_Co[i][33])
  minimat(19,9,22,10,hea_vac_Co[i][34])
  minimat(19,7,22,8,hea_vac_Co[i][35])

  minimat(17,3,18,6,hea_vac_Co[i][36])
  minimat(15,3,16,6,hea_vac_Co[i][37])
  minimat(13,3,14,6,hea_vac_Co[i][38])
  minimat(11,3,12,6,hea_vac_Co[i][39])
  minimat(9,3,10,6,hea_vac_Co[i][40])
  minimat(7,3,8,6,hea_vac_Co[i][41])

  minimat(1,7,2,12,hea_vac_Co[i][42])
  minimat(1,13,2,18,hea_vac_Co[i][43])

  minimat(7,23,12,24,hea_vac_Co[i][44])
  minimat(13,23,18,24,hea_vac_Co[i][45])

  minimat(23,13,24,18,hea_vac_Co[i][46])
  minimat(23,7,24,12,hea_vac_Co[i][47])

  minimat(13,1,18,2,hea_vac_Co[i][48])
  minimat(7,1,12,2,hea_vac_Co[i][49])

  minimat(1,1,4,2,hea_vac_Co[i][50])
  minimat(1,3,2,4,hea_vac_Co[i][50])

  minimat(1,23,4,24,hea_vac_Co[i][51])
  minimat(1,21,2,22,hea_vac_Co[i][51])

  minimat(21,23,24,24,hea_vac_Co[i][52])
  minimat(23,21,24,22,hea_vac_Co[i][52])

  minimat(21,1,24,2,hea_vac_Co[i][53])
  minimat(23,3,24,4,hea_vac_Co[i][53])
  datasetCo.append(newimg)

print(hea_vac_Cr[0])

datasetCr = []
for i in range(len(hea_vac_Cr)):
  newimg = np.zeros((24,24),dtype = int)
  def minimat(inicordx,inicordy,finalcordx,finalcordy,heaval):
    for i in range(inicordx-1,finalcordx):
      for j in range(inicordy-1,finalcordy):
        newimg[i][j] = heaval
    return newimg
  minimat(11,11,14,14,hea_vac_Cr[0][54])
  minimat(9,9,10,10,hea_vac_Cr[i][0])
  minimat(9,11,10,12,hea_vac_Cr[i][1])
  minimat(9,13,10,14,hea_vac_Cr[i][2])
  minimat(9,15,10,16,hea_vac_Cr[i][3])
  minimat(11,15,12,16,hea_vac_Cr[i][4])
  minimat(13,15,14,16,hea_vac_Cr[i][5])
  minimat(15,15,16,16,hea_vac_Cr[i][6])  
  minimat(15,13,16,14,hea_vac_Cr[i][7])
  minimat(15,11,16,12,hea_vac_Cr[i][8])
  minimat(15,9,16,10,hea_vac_Cr[i][9])
  minimat(13,9,14,10,hea_vac_Cr[i][10])
  minimat(11,9,12,10,hea_vac_Cr[i][11])
  minimat(7,11,8,14,hea_vac_Cr[i][12])
  minimat(11,17,14,18,hea_vac_Cr[i][13])
  minimat(17,11,18,14,hea_vac_Cr[i][14])
  minimat(11,7,14,8,hea_vac_Cr[i][15])

  minimat(7,7,8,8,hea_vac_Cr[i][16])
  minimat(17,17,18,18,hea_vac_Cr[i][16])

  minimat(7,17,8,18,hea_vac_Cr[i][17])
  minimat(17,7,18,8,hea_vac_Cr[i][17])

  minimat(3,7,6,8,hea_vac_Cr[i][18])
  minimat(3,9,6,10,hea_vac_Cr[i][19])
  minimat(3,11,6,12,hea_vac_Cr[i][20])
  minimat(3,13,6,14,hea_vac_Cr[i][21])
  minimat(3,15,6,16,hea_vac_Cr[i][22])
  minimat(3,17,6,18,hea_vac_Cr[i][23])

  minimat(7,19,8,22,hea_vac_Cr[i][24])
  minimat(9,19,10,22,hea_vac_Cr[i][25])
  minimat(11,19,12,22,hea_vac_Cr[i][26])
  minimat(13,19,14,22,hea_vac_Cr[i][27])
  minimat(15,19,16,22,hea_vac_Cr[i][28])
  minimat(17,19,18,22,hea_vac_Cr[i][29])

  minimat(19,17,22,18,hea_vac_Cr[i][30])
  minimat(19,15,22,16,hea_vac_Cr[i][31])
  minimat(19,13,22,14,hea_vac_Cr[i][32])
  minimat(19,11,22,12,hea_vac_Cr[i][33])
  minimat(19,9,22,10,hea_vac_Cr[i][34])
  minimat(19,7,22,8,hea_vac_Cr[i][35])

  minimat(17,3,18,6,hea_vac_Cr[i][36])
  minimat(15,3,16,6,hea_vac_Cr[i][37])
  minimat(13,3,14,6,hea_vac_Cr[i][38])
  minimat(11,3,12,6,hea_vac_Cr[i][39])
  minimat(9,3,10,6,hea_vac_Cr[i][40])
  minimat(7,3,8,6,hea_vac_Cr[i][41])

  minimat(1,7,2,12,hea_vac_Cr[i][42])
  minimat(1,13,2,18,hea_vac_Cr[i][43])

  minimat(7,23,12,24,hea_vac_Cr[i][44])
  minimat(13,23,18,24,hea_vac_Cr[i][45])

  minimat(23,13,24,18,hea_vac_Cr[i][46])
  minimat(23,7,24,12,hea_vac_Cr[i][47])

  minimat(13,1,18,2,hea_vac_Cr[i][48])
  minimat(7,1,12,2,hea_vac_Cr[i][49])

  minimat(1,1,4,2,hea_vac_Cr[i][50])
  minimat(1,3,2,4,hea_vac_Cr[i][50])

  minimat(1,23,4,24,hea_vac_Cr[i][51])
  minimat(1,21,2,22,hea_vac_Cr[i][51])

  minimat(21,23,24,24,hea_vac_Cr[i][52])
  minimat(23,21,24,22,hea_vac_Cr[i][52])

  minimat(21,1,24,2,hea_vac_Cr[i][53])
  minimat(23,3,24,4,hea_vac_Cr[i][53])
  datasetCr.append(newimg)



datasetFe = []
for i in range(len(hea_vac_Fe)):
  newimg = np.zeros((24,24),dtype = int)
  def minimat(inicordx,inicordy,finalcordx,finalcordy,heaval):
    for i in range(inicordx-1,finalcordx):
      for j in range(inicordy-1,finalcordy):
        newimg[i][j] = heaval
    return newimg
  minimat(11,11,14,14,hea_vac_Fe[0][54])
  minimat(9,9,10,10,hea_vac_Fe[i][0])
  minimat(9,11,10,12,hea_vac_Fe[i][1])
  minimat(9,13,10,14,hea_vac_Fe[i][2])
  minimat(9,15,10,16,hea_vac_Fe[i][3])
  minimat(11,15,12,16,hea_vac_Fe[i][4])
  minimat(13,15,14,16,hea_vac_Fe[i][5])
  minimat(15,15,16,16,hea_vac_Fe[i][6])  
  minimat(15,13,16,14,hea_vac_Fe[i][7])
  minimat(15,11,16,12,hea_vac_Fe[i][8])
  minimat(15,9,16,10,hea_vac_Fe[i][9])
  minimat(13,9,14,10,hea_vac_Fe[i][10])
  minimat(11,9,12,10,hea_vac_Fe[i][11])
  minimat(7,11,8,14,hea_vac_Fe[i][12])
  minimat(11,17,14,18,hea_vac_Fe[i][13])
  minimat(17,11,18,14,hea_vac_Fe[i][14])
  minimat(11,7,14,8,hea_vac_Fe[i][15])

  minimat(7,7,8,8,hea_vac_Fe[i][16])
  minimat(17,17,18,18,hea_vac_Fe[i][16])

  minimat(7,17,8,18,hea_vac_Fe[i][17])
  minimat(17,7,18,8,hea_vac_Fe[i][17])

  minimat(3,7,6,8,hea_vac_Fe[i][18])
  minimat(3,9,6,10,hea_vac_Fe[i][19])
  minimat(3,11,6,12,hea_vac_Fe[i][20])
  minimat(3,13,6,14,hea_vac_Fe[i][21])
  minimat(3,15,6,16,hea_vac_Fe[i][22])
  minimat(3,17,6,18,hea_vac_Fe[i][23])

  minimat(7,19,8,22,hea_vac_Fe[i][24])
  minimat(9,19,10,22,hea_vac_Fe[i][25])
  minimat(11,19,12,22,hea_vac_Fe[i][26])
  minimat(13,19,14,22,hea_vac_Fe[i][27])
  minimat(15,19,16,22,hea_vac_Fe[i][28])
  minimat(17,19,18,22,hea_vac_Fe[i][29])

  minimat(19,17,22,18,hea_vac_Fe[i][30])
  minimat(19,15,22,16,hea_vac_Fe[i][31])
  minimat(19,13,22,14,hea_vac_Fe[i][32])
  minimat(19,11,22,12,hea_vac_Fe[i][33])
  minimat(19,9,22,10,hea_vac_Fe[i][34])
  minimat(19,7,22,8,hea_vac_Fe[i][35])

  minimat(17,3,18,6,hea_vac_Fe[i][36])
  minimat(15,3,16,6,hea_vac_Fe[i][37])
  minimat(13,3,14,6,hea_vac_Fe[i][38])
  minimat(11,3,12,6,hea_vac_Fe[i][39])
  minimat(9,3,10,6,hea_vac_Fe[i][40])
  minimat(7,3,8,6,hea_vac_Fe[i][41])

  minimat(1,7,2,12,hea_vac_Fe[i][42])
  minimat(1,13,2,18,hea_vac_Fe[i][43])

  minimat(7,23,12,24,hea_vac_Fe[i][44])
  minimat(13,23,18,24,hea_vac_Fe[i][45])

  minimat(23,13,24,18,hea_vac_Fe[i][46])
  minimat(23,7,24,12,hea_vac_Fe[i][47])

  minimat(13,1,18,2,hea_vac_Fe[i][48])
  minimat(7,1,12,2,hea_vac_Fe[i][49])

  minimat(1,1,4,2,hea_vac_Fe[i][50])
  minimat(1,3,2,4,hea_vac_Fe[i][50])

  minimat(1,23,4,24,hea_vac_Fe[i][51])
  minimat(1,21,2,22,hea_vac_Fe[i][51])

  minimat(21,23,24,24,hea_vac_Fe[i][52])
  minimat(23,21,24,22,hea_vac_Fe[i][52])

  minimat(21,1,24,2,hea_vac_Fe[i][53])
  minimat(23,3,24,4,hea_vac_Fe[i][53])
  datasetFe.append(newimg)

datasetNi = []
for i in range(len(hea_vac_Ni)):
  newimg = np.zeros((24,24),dtype = int)
  def minimat(inicordx,inicordy,finalcordx,finalcordy,heaval):
    for i in range(inicordx-1,finalcordx):
      for j in range(inicordy-1,finalcordy):
        newimg[i][j] = heaval
    return newimg
  minimat(11,11,14,14,hea_vac_Ni[0][54])
  minimat(9,9,10,10,hea_vac_Ni[i][0])
  minimat(9,11,10,12,hea_vac_Ni[i][1])
  minimat(9,13,10,14,hea_vac_Ni[i][2])
  minimat(9,15,10,16,hea_vac_Ni[i][3])
  minimat(11,15,12,16,hea_vac_Ni[i][4])
  minimat(13,15,14,16,hea_vac_Ni[i][5])
  minimat(15,15,16,16,hea_vac_Ni[i][6])  
  minimat(15,13,16,14,hea_vac_Ni[i][7])
  minimat(15,11,16,12,hea_vac_Ni[i][8])
  minimat(15,9,16,10,hea_vac_Ni[i][9])
  minimat(13,9,14,10,hea_vac_Ni[i][10])
  minimat(11,9,12,10,hea_vac_Ni[i][11])
  minimat(7,11,8,14,hea_vac_Ni[i][12])
  minimat(11,17,14,18,hea_vac_Ni[i][13])
  minimat(17,11,18,14,hea_vac_Ni[i][14])
  minimat(11,7,14,8,hea_vac_Ni[i][15])

  minimat(7,7,8,8,hea_vac_Ni[i][16])
  minimat(17,17,18,18,hea_vac_Ni[i][16])

  minimat(7,17,8,18,hea_vac_Ni[i][17])
  minimat(17,7,18,8,hea_vac_Ni[i][17])

  minimat(3,7,6,8,hea_vac_Ni[i][18])
  minimat(3,9,6,10,hea_vac_Ni[i][19])
  minimat(3,11,6,12,hea_vac_Ni[i][20])
  minimat(3,13,6,14,hea_vac_Ni[i][21])
  minimat(3,15,6,16,hea_vac_Ni[i][22])
  minimat(3,17,6,18,hea_vac_Ni[i][23])

  minimat(7,19,8,22,hea_vac_Ni[i][24])
  minimat(9,19,10,22,hea_vac_Ni[i][25])
  minimat(11,19,12,22,hea_vac_Ni[i][26])
  minimat(13,19,14,22,hea_vac_Ni[i][27])
  minimat(15,19,16,22,hea_vac_Ni[i][28])
  minimat(17,19,18,22,hea_vac_Ni[i][29])

  minimat(19,17,22,18,hea_vac_Ni[i][30])
  minimat(19,15,22,16,hea_vac_Ni[i][31])
  minimat(19,13,22,14,hea_vac_Ni[i][32])
  minimat(19,11,22,12,hea_vac_Ni[i][33])
  minimat(19,9,22,10,hea_vac_Ni[i][34])
  minimat(19,7,22,8,hea_vac_Ni[i][35])

  minimat(17,3,18,6,hea_vac_Ni[i][36])
  minimat(15,3,16,6,hea_vac_Ni[i][37])
  minimat(13,3,14,6,hea_vac_Ni[i][38])
  minimat(11,3,12,6,hea_vac_Ni[i][39])
  minimat(9,3,10,6,hea_vac_Ni[i][40])
  minimat(7,3,8,6,hea_vac_Ni[i][41])

  minimat(1,7,2,12,hea_vac_Ni[i][42])
  minimat(1,13,2,18,hea_vac_Ni[i][43])

  minimat(7,23,12,24,hea_vac_Ni[i][44])
  minimat(13,23,18,24,hea_vac_Ni[i][45])

  minimat(23,13,24,18,hea_vac_Ni[i][46])
  minimat(23,7,24,12,hea_vac_Ni[i][47])

  minimat(13,1,18,2,hea_vac_Ni[i][48])
  minimat(7,1,12,2,hea_vac_Ni[i][49])

  minimat(1,1,4,2,hea_vac_Ni[i][50])
  minimat(1,3,2,4,hea_vac_Ni[i][50])

  minimat(1,23,4,24,hea_vac_Ni[i][51])
  minimat(1,21,2,22,hea_vac_Ni[i][51])

  minimat(21,23,24,24,hea_vac_Ni[i][52])
  minimat(23,21,24,22,hea_vac_Ni[i][52])

  minimat(21,1,24,2,hea_vac_Ni[i][53])
  minimat(23,3,24,4,hea_vac_Ni[i][53])
  datasetNi.append(newimg)



dataset = []
for i in range(len(datasetCo)):
  datasetCo[i] = [datasetCo[i],Covfe[i][55]]
for i in range(len(datasetCr)):
  datasetCr[i] = [datasetCr[i],Crvfe[i][55]]
for i in range(len(datasetFe)):
  datasetFe[i] = [datasetFe[i],Fevfe[i][55]]
for i in range(len(datasetNi)):
  datasetNi[i] = [datasetNi[i],Nivfe[i][55]]


dataset = datasetCo+datasetCr+datasetFe+datasetNi
dataset = np.array(dataset)

np.random.shuffle(dataset)

print(dataset[0][0])

train = dataset[0:4210]
valid = dataset[4210:5112]
test = dataset[5112:6015]



trainx = []
trainy = []
validx = []
validy = []
testx = []
testy = []
for i in range(len(train)):
  trainx.append(train[i][0])
  trainy.append(train[i][1])
for i in range(len(valid)):
  validx.append(valid[i][0])
  validy.append(valid[i][1])
for i in range(len(test)):
  testx.append(test[i][0])
  testy.append(test[i][1])

trainx = np.array(trainx)
trainx = trainx.reshape(-1,24,24,1)
trainx.shape
trainy = np.array(trainy)
validy = np.array(validy)
testy = np.array(testy)

validx = np.array(validx)

validx = validx.reshape(-1,24,24,1)


testx = np.array(testx)
testx = testx.reshape(-1,24,24,1)

print(len(testx))
print(len(testy))

for i in range(len(trainx)):
  for j in range(0,24):
    for k in range(0,24):
      trainx[i][j][k][0] = trainx[i][j][k][0]/4

for i in range(len(validx)):
  for j in range(0,24):
    for k in range(0,24):
      validx[i][j][k][0] = validx[i][j][k][0]/4
for i in range(len(testx)):
  for j in range(len(testx[i])):
    for k in range(len(testx[i][j])):
      testx[i][j][k][0] = testx[i][j][k][0]/4

model = models.Sequential()
model.add(layers.RandomFlip("horizontal_and_vertical"))
model.add(layers.Conv2D(32,(3,3),activation = 'relu' ,padding = "same", input_shape = (24,24,1),kernel_initializer=initializers.RandomNormal(stddev=0.01),bias_initializer=initializers.Zeros()))

model.add(layers.AveragePooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu' ))

model.add(layers.AveragePooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation = 'relu' ))
model.add(layers.Conv2D(128,(3,3),activation = 'relu' ))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(1))

adam = keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer = adam, loss = tf.keras.losses.MeanSquaredError())
history = model.fit(trainx,trainy,batch_size = 32, epochs = 100,validation_data=(validx,validy))

trainloss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(trainloss)+1)

import matplotlib.pyplot as plt
plt.plot(epochs,trainloss, label = "Training Loss")
plt.plot(epochs,val_loss, label = "Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
leg = plt.legend(loc='upper center')
plt.show

adam = keras.optimizers.Adam(learning_rate = 0.0001)
model.compile(optimizer = adam, loss = tf.keras.losses.MeanSquaredError())
history = model.fit(trainx,trainy,batch_size = 32, epochs = 20,validation_data=(validx,validy))

adam = keras.optimizers.Adam(learning_rate = 0.000001)
model.compile(optimizer = adam, loss = tf.keras.losses.MeanSquaredError())
history = model.fit(trainx,trainy,batch_size = 32, epochs = 20,validation_data=(validx,validy))

import math
x = model.predict(testx)

pred = []
for i in range(len(x)):
  pred.append(float(x[i][0]))

numcorrect = 0
prederror = []
for i in range(len(pred)):
  prederror.append(abs(pred[i]-testy[i]))
  if (prederror[i])<0.05:
    numcorrect += 1
accuracy = numcorrect/len(testy)
print(accuracy)
RMSE = math.sqrt(val_loss[-1])
print(RMSE)

plt.scatter(pred,testy)
plt.xlabel("Predicted Value")
plt.ylabel("True Value")
plt.show