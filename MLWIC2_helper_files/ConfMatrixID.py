import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
import matplotlib.patches as patches
from cStringIO import StringIO 
import sys
import argparse

matplotlib.rcParams['figure.figsize'] = (16.0, 9.0)
names_dic={0:'Moose',1:'Cow',2:'Quail',3:'Dog',4:'Elk',5:'Red Deer',6:'Corvid',7:'Armadillo',8:'Galliformes',9:'Opossum',10:'Horse',11:'Human',12:'Rabbit',13:'Lynx',14:'Skunk',15:'Unidentfied Deer',16:'Small Mammal',17:'Mule Deer',18:'White-tailed Deer',19:'Raccoon',20:'Mountain Lion',21:'Squirrel',22:'Wild Pig',23:'Fox',24:'Black Bear',25:'Truck',26:'Bird',27:'Empty'}

parser = argparse.ArgumentParser(description='Process Command-line Arguments')
parser.add_argument('predictions_file', type= str, action= 'store', help= 'The predictions info file')
parser.add_argument('--filter', nargs=2, default= [-1,-1], type= int, action= 'store', help= 'Labels to filter predictions')
args = parser.parse_args()

filename = args.predictions_file
num_classes = 28
file_contents = open(filename, 'r').read() 
#paths=np.loadtxt(StringIO(file_contents.replace('[', '').replace(']', '')),usecols=[1],delimiter=',',dtype='str')
#preds=np.loadtxt(StringIO(file_contents.replace('[', '').replace(']', '')),usecols=(2,3),delimiter=',')
paths=np.loadtxt(io.StringIO(file_contents.replace('[', '').replace(']', '')),usecols=[1],delimiter=',',dtype='str')
preds=np.loadtxt(io.StringIO(file_contents.replace('[', '').replace(']', '')),usecols=(2,3),delimiter=',')

conf_arr= np.zeros((num_classes,num_classes))
print(preds.shape)
if(args.filter[0]!=-1 and args.filter[1]!=-1):
  for i in range(0,preds.shape[0]):
    conf_arr[int(preds[i,0]),int(preds[i,1])]+=1
    if preds[i,0]==args.filter[0] and preds[i,1]==args.filter[1]:
      print(paths[i])
else:
  for i in range(0,preds.shape[0]):
    conf_arr[int(preds[i,0]),int(preds[i,1])]+=1

norm_conf = conf_arr/conf_arr.sum(axis=1, keepdims=True)
norm_conf[np.isnan(norm_conf)]=0
#norm_conf.fill(0)


fig = plt.figure(figsize=(20,20))
plt.clf()
ax = fig.add_subplot(111)
ax.invert_yaxis()
ax.set_xticks([int(x)*2+0.5 for x in names_dic.keys()])
ax.set_xticklabels(names_dic.values(), rotation=90, fontweight='bold')
ax.set_yticks([int(x)*2+0.5 for x in names_dic.keys()])
ax.set_yticklabels(names_dic.values(), fontweight='bold')
ax.set_xticks([int(x)*2+1.5 for x in names_dic.keys()], minor=True)
ax.set_yticks([int(x)*2+1.5 for x in names_dic.keys()], minor=True)
ax.grid(b=None, which='minor', axis='both', ls="solid")
ax.set_xlabel('Model Predictions', fontsize=18, fontweight='bold')
ax.set_ylabel('Ground Truth Labels', fontsize=18, fontweight='bold')
ex_norm_conf= np.zeros((norm_conf.shape[0]*2,norm_conf.shape[1]*2))
#ex_norm_conf[:,0:96:2]= norm_conf
#ex_norm_conf[:,1:96:2]= norm_conf
res = ax.imshow(np.array(ex_norm_conf), cmap=plt.cm.Blues)#, interpolation='nearest')

width, height = conf_arr.shape

plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.25)
for x in xrange(width):
  for y in xrange(height):
    if int(conf_arr[x][y])!=0:
      ax.add_patch(patches.Rectangle( (2*y-0.5,2*x-0.5), 2, 2, facecolor="#00ff00" if x==y else "#ff0000", alpha= norm_conf[x][y], zorder=2))
      ax.annotate(str(int(conf_arr[x][y])), xy=(2*y+0.5, 2*x+0.5), horizontalalignment='center', verticalalignment='center', annotation_clip=True, fontsize=8, fontweight='bold')

#cb = fig.colorbar(res)
#plt.show()
plt.savefig("ConfMatID.pdf")
