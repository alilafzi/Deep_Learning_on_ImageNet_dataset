import argparse
parser = argparse . ArgumentParser ( description = 'HW02 Task2'
)
parser . add_argument ( '--imagenet_root' , type = str , required
= True )
parser . add_argument ( '--class_list' , nargs = '*' , type = str ,
required = True )
args , args_other = parser . parse_known_args ()
#print (args.class_list)

import torch
from torch . utils . data import DataLoader , Dataset
import scipy
from scipy import misc
import os
import glob
import PIL
from PIL import Image
import numpy as np
from skimage import io, transform
import torch.nn.functional as F

class your_dataset_class ( Dataset ) :
	def __init__ (self, imagenet_root, class_list, transform, goal) :
		self.path = imagenet_root
		self.classes = class_list
		self.transform = transform 
		self.goal = goal
		main_directory = os.path.join(self.path, self.goal)
		self.x_data = []
		self.y_data = []
		for i in range(0, len(self.classes)):
			os.chdir(main_directory)
			os.chdir(self.classes[i])
			label = torch.zeros(len(self.classes))
			label[i] = 1
			folders = os.listdir(".")
			for sub in folders:
				os.chdir(sub)
				files = os.listdir(os.getcwd())
				for filename in files:	
					self.x_data.append(self.transform(Image.open(filename)))
					self.y_data.append(label)
				os.chdir('..')
		self.len = len(self.x_data)

	def __getitem__ (self, index) :
		#print (self.x_data[index], self.y_data[index])
		return self.x_data[index], self.y_data[index] 

	def __len__ (self):
		#print (self.len)
		return self.len

#if __name__ == "__main__":
	#your_dataset_class ( args.imagenet_root, args.class_list ).__len__()

from torchvision import transforms as tvt
transform = tvt . Compose ( [ tvt . ToTensor () , tvt . Normalize (( 0.5 ,0.5 ,0.5 ) , ( 0.5 , 0.5 , 0.5 ) ) ] )
train_dataset = your_dataset_class (args.imagenet_root, args.class_list, transform, 'Train')
#print ('train size', train_dataset.__len__())
train_data_loader = torch.utils.data.DataLoader(dataset =
train_dataset ,
batch_size = 10 ,
shuffle = True ,
num_workers = 4 )

val_dataset = your_dataset_class (args.imagenet_root, args.class_list, transform, 'Val')
#print ('val size', val_dataset.__len__())
val_data_loader = torch.utils.data.DataLoader(dataset =
val_dataset ,
batch_size = val_dataset.__len__() ,
shuffle = True ,
num_workers = 4 )

import torch
import random

# TODO Follow the recommendations from the lecture notes to ensure reproducible results
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.becnhmarks = False
os.environ['PYTHONHASHSEED'] = str(seed)

dtype = torch . float64

device = torch . device ( "cuda:0" if torch.cuda.is_available() else "cpu" )

##Training
epochs = 30 # feel free to adjust this parameter
D_in , H1 , H2 , D_out = 3 * 64 * 64 , 1000 , 256 , 2
w1 = torch . randn ( D_in , H1 , device = device , dtype = dtype )
w2 = torch . randn ( H1 , H2 , device = device , dtype = dtype )
w3 = torch . randn ( H2 , D_out , device = device , dtype = dtype )
learning_rate = 1e-9

os.chdir(args.imagenet_root)
for t in range ( epochs ) :
	epoch_loss = 0
	
	for i , data in enumerate (train_data_loader) :
		inputs , labels = data
		inputs = inputs . to ( device ).double()
		labels = labels . to ( device )
		#inputs.dtype = dtype
		x = inputs . view ( inputs . size ( 0 ) , - 1)
		#print (inputs.dtype)
		#print (w1.dtype)
		#print (x.dtype)
		h1 = x . mm ( w1 )
		##numpy , you would say. dot ( w1 )
		h1_relu = h1 . clamp ( min = 0 )
		h2 = h1_relu . mm ( w2 )
		h2_relu = h2 . clamp ( min = 0 )
		y_pred = h2_relu . mm ( w3 )
		# Compute and print loss
		y = labels
		loss = ( y_pred - y ) . pow ( 2 ) . sum () . item ()
		y_error = y_pred - y
		

		# TODO : Accumulate loss for printing per epoch
		epoch_loss += loss

		grad_w3 = h2_relu . t () . mm ( 2 * y_error )
		
		h2_error = 2.0 * y_error . mm ( w3 . t () )
		#backpropagated error to the h2 hidden layer
		h2_error [ h2 < 0 ] = 0
		# We set those elements of the backpropagated error
		grad_w2 = h1_relu . t () . mm ( 2 * h2_error ) # <<<<<<Gradient of Loss w . r . t w2
		h1_error = 2.0 * h2_error . mm ( w2 . t () ) #backpropagated error to the h1 hidden layer
		h1_error [ h1 < 0 ] = 0
		# We set those elements of the backpropagated error
		grad_w1 = x . t () . mm ( 2 * h1_error )# <<<<<<Gradient of Loss w . r . t w2
		# Update weights using gradient descent
		w1 -= learning_rate * grad_w1
		w2 -= learning_rate * grad_w2
		w3 -= learning_rate * grad_w3
		
	# print loss per epoch
	if t==0:
		with open('output.txt', 'w') as f:
			print ( 'Epoch %d:\t %0.4f'%(t , epoch_loss/len(y) ) , file = f)
	else:
		with open('output.txt', 'a') as f:
			print ( 'Epoch %d:\t %0.4f'%(t , epoch_loss/len(y) ) , file = f)

#print (y)
#print (y_pred)
# Store layer weights in pickle file format
os.chdir(args.imagenet_root)
torch.save({'w1':w1 , 'w2':w2 , 'w3':w3} , './wts.pkl')

##Validation
import pickle
with open('wts.pkl', 'rb') as infile:
	weights_dict = torch.load(infile)
infile.close()

#print (weights_dict)
w1 = weights_dict['w1']
w2 = weights_dict['w2']
w3 = weights_dict['w3']

for i , data in enumerate (val_data_loader) :
		inputs , labels = data
		inputs = inputs . to ( device ).double()
		labels = labels . to ( device )
		#inputs.dtype = dtype
		x = inputs . view ( inputs . size ( 0 ) , - 1)
		h1 = x . mm ( w1 )
		##numpy , you would say. dot ( w1 )
		h1_relu = h1 . clamp ( min = 0 )
		h2 = h1_relu . mm ( w2 )
		h2_relu = h2 . clamp ( min = 0 )
		y_pred = h2_relu . mm ( w3 )
		# Compute and print loss
		y = labels
		loss = ( y_pred - y ) . pow ( 2 ) . sum () . item ()
		y_error = y_pred - y

def accuracy(y_pred, y):
	count = 0
	for i in range(len(y)):
		if torch.argmax(y_pred[i]) == torch.argmax(y[i]):
			count += 1

	return count/len(y)*100
#print (accuracy(y_pred, y))
#print (y)
#print (y_pred)
with open('output.txt', 'a') as f:
			f.write('\n')
			print ( 'Val Loss:\t %0.4f'%( loss/len(y) ) , file = f)
			print ( 'Val Accuracy:\t %0.4f'%( accuracy(y_pred,y))+'%', file = f) 
