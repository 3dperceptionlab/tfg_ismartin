#Prueba Yolo
#Step1: Import libraries
import os
import scipy.io
import scipy.misc
import numpy as np
#import pandas as pd
import PIL
from PIL import Image
import struct
#Necesario hacer:
'''
cd interactiveObjectsYolo/
pip install opencv-python
apt update
apt install libgl1-mesa-glx -y
python main.py
'''
import cv2
from numpy import expand_dims
import tensorflow as tf
#from skimage.transform import resize
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.models import load_model, Model
from keras.layers.merge import add, concatenate
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.patches import Rectangle
#%matplotlib inline
import csv

from WeightReader import WeightReader
from BoundBox import BoundBox

#Step2: Read yolov3.weights file (Class WeightReader)
WEIGHTS_FILE='../../models/yolov3.weights'	#COCO Dataset pretrained weights

#Step3: Create yolo neural network
def _conv_block(inp, convs, skip=True):
	x = inp
	count = 0
	for conv in convs:
		if count == (len(convs) - 2) and skip:
			skip_connection = x
		count += 1
		if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top
		x = Conv2D(conv['filter'],
				   conv['kernel'],
				   strides=conv['stride'],
				   padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
				   name='conv_' + str(conv['layer_idx']),
				   use_bias=False if conv['bnorm'] else True)(x)
		if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
		if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)
	return add([skip_connection, x]) if skip else x

def make_yolov3_model():
	input_image = Input(shape=(None, None, 3))
	# Layer  0 => 4
	x = _conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
								  {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
								  {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
								  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])
	# Layer  5 => 8
	x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
						{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
						{'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])
	# Layer  9 => 11
	x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
						{'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])
	# Layer 12 => 15
	x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
						{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
						{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])
	# Layer 16 => 36
	for i in range(7):
		x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
							{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])
	skip_36 = x
	# Layer 37 => 40
	x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
						{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
						{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])
	# Layer 41 => 61
	for i in range(7):
		x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
							{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])
	skip_61 = x
	# Layer 62 => 65
	x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
						{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
						{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])
	# Layer 66 => 74
	for i in range(3):
		x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
							{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])
	# Layer 75 => 79
	x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
						{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
						{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
						{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
						{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], skip=False)
	# Layer 80 => 82
	yolo_82 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
							  {'filter':  255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], skip=False)
	# Layer 83 => 86
	x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], skip=False)
	x = UpSampling2D(2)(x)
	x = concatenate([x, skip_61])
	# Layer 87 => 91
	x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
						{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
						{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
						{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
						{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], skip=False)
	# Layer 92 => 94
	yolo_94 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
							  {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}], skip=False)
	# Layer 95 => 98
	x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], skip=False)
	x = UpSampling2D(2)(x)
	x = concatenate([x, skip_36])
	# Layer 99 => 106
	yolo_106 = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
							   {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
							   {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
							   {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
							   {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
							   {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},
							   {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}], skip=False)
	model = Model(input_image, [yolo_82, yolo_94, yolo_106])
	return model

#Step4: Use previous functions

# define the yolo v3 model
yolov3 = make_yolov3_model()

# load the weights
weight_reader = WeightReader(WEIGHTS_FILE)

# set the weights
weight_reader.load_weights(yolov3)

# save the model to file
yolov3.save('model.h5')

#Step5: Decode prediction outputs (BoundBox class)


def _sigmoid(x):
  return 1. /(1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
	grid_h, grid_w = netout.shape[:2]
	nb_box = 3
	netout = netout.reshape((grid_h, grid_w, nb_box, -1))
	nb_class = netout.shape[-1] - 5
	boxes = []
	netout[..., :2]  = _sigmoid(netout[..., :2])
	netout[..., 4:]  = _sigmoid(netout[..., 4:])
	netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
	netout[..., 5:] *= netout[..., 5:] > obj_thresh
 
	for i in range(grid_h*grid_w):
		row = i / grid_w
		col = i % grid_w
		for b in range(nb_box):
			# 4th element is objectness score
			objectness = netout[int(row)][int(col)][b][4]
			if(objectness.all() <= obj_thresh): continue
			# first 4 elements are x, y, w, and h
			x, y, w, h = netout[int(row)][int(col)][b][:4]
			x = (col + x) / grid_w # center position, unit: image width
			y = (row + y) / grid_h # center position, unit: image height
			w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
			h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
			# last elements are class probabilities
			classes = netout[int(row)][col][b][5:]

			box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
			boxes.append(box)
	return boxes

#Step6: Scale and stretch boxes
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
	new_w, new_h = net_w, net_h
	for i in range(len(boxes)):
		x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
		y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
		boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
		boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
		boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
		boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

#Step7: Avoid repeated objects
def _interval_overlap(interval_a, interval_b):
	x1, x2 = interval_a
	x3, x4 = interval_b
	if x3 < x1:
		if x4 < x1:
			return 0
		else:
			return min(x2,x4) - x1
	else:
		if x2 < x3:
			 return 0
		else:
			return min(x2,x4) - x3
 
def bbox_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
	intersect = intersect_w * intersect_h
	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	union = w1*h1 + w2*h2 - intersect
	return float(intersect) / union
 
def do_nms(boxes, nms_thresh):
	if len(boxes) > 0:
		nb_class = len(boxes[0].classes)
	else:
		return
	for c in range(nb_class):
		sorted_indices = np.argsort([-box.classes[c] for box in boxes])
		for i in range(len(sorted_indices)):
			index_i = sorted_indices[i]
			if boxes[index_i].classes[c] == 0: continue
			for j in range(i+1, len(sorted_indices)):
				index_j = sorted_indices[j]
				if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
					boxes[index_j].classes[c] = 0

# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
	v_boxes, v_labels, v_scores, v_index = list(), list(), list(), list()
	# enumerate all boxes
	for box in boxes:
		# enumerate all possible labels
		for i in range(len(labels)):
			# check if the threshold for this label is high enough
			# check if it is an interactionable object
			if box.classes[i] > thresh and actionsObjects[i]:
				v_boxes.append(box)
				v_labels.append(labels[i])
				v_scores.append(box.classes[i]*100)
				v_index.append(i)
				# don't break, many labels may trigger for one box
	return v_boxes, v_labels, v_scores, v_index

# get all of the results above a threshold
def get_boxes_light(boxes, labels, thresh):
	v_labels, v_index = list(), list()
	# enumerate all boxes
	for box in boxes:
		# enumerate all possible labels
		for i in range(len(labels)):
			# check if the threshold for this label is high enough
			# check if it is an interactionable object
			if actionsObjects[i] and box.classes[i] > thresh:
				v_labels.append(labels[i])
				v_index.append(i)
				# don't break, many labels may trigger for one box
	return v_labels, v_index

# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
  
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for i in range(len(v_boxes)):
		box = v_boxes[i]
		# get coordinates
		y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
		# calculate width and height of the box
		width, height = x2 - x1, y2 - y1
		# create the shape
		rect = Rectangle((x1, y1), width, height, fill=False, color='yellow', linewidth = '2')
		# draw the box
		ax.add_patch(rect)
		# draw text and score in top left corner
		label = "%s (%.3f)" % (v_labels[i], v_scores[i])
		pyplot.text(x1, y1, label, color='yellow')
	# show the plot
	pyplot.show()
	# save the plot
	pyplot.savefig('outputs/prueba.png')

#Step8: Declare anchors,probability threshold and labels
# define the anchors
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

# define the probability threshold for detected objects
class_threshold = 0.4

# define the labels
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
	"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# define actions objects
# Actualmente se realiza manualmente, tras obtener datos del dataset
# se realizara de forma automatica

# Falta definir acciones para:
# "bicycle", "tie", "frisbee", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
# 	"tennis racket",
# 	"pottedplant", "toilet",
# 	"remote", "microwave", "oven", "toaster", "sink", "refrigerator",
# 	"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"

actionsToIgnore = ["none","n","","on","no","na","touch","use","no action"]
actionsInToyota = ["maketea","laydown","usetablet","cook","enter","walk","getup","drink","usetelephone",
"readbook","eat","uselaptop","watchtv","leave","sitdown","makecoffee","cutbread","pour","takepills"] #19 acciones
#Quitando walk, enter y leave quedan 16 acciones posibles

#actionsToIgnore = []
def defineActionsObjects():
	actionsObjects = [set() for _ in range(len(labels))]

	'''actionsObjects[labels.index("book")]={"read","readbook"}

	actionsObjects[labels.index("refrigerator")]={"eat","cook"}
	actionsObjects[labels.index("microwave")]={"eat","cook"}
	actionsObjects[labels.index("toaster")]={"eat","cook"}

	actionsObjects[labels.index("fork")]={"eat"}
	actionsObjects[labels.index("knife")]=set(actionsObjects[labels.index("fork")])
	actionsObjects[labels.index("spoon")]=set(actionsObjects[labels.index("fork")])
	actionsObjects[labels.index("bowl")]={"eat","pour"}

	actionsObjects[labels.index("banana")]={"eat"}
	actionsObjects[labels.index("apple")]=set(actionsObjects[labels.index("banana")])
	actionsObjects[labels.index("sandwich")]=set(actionsObjects[labels.index("banana")])
	actionsObjects[labels.index("broccoli")]=set(actionsObjects[labels.index("banana")])
	actionsObjects[labels.index("carrot")]=set(actionsObjects[labels.index("banana")])
	actionsObjects[labels.index("hot dog")]=set(actionsObjects[labels.index("banana")])
	actionsObjects[labels.index("pizza")]=set(actionsObjects[labels.index("banana")])
	actionsObjects[labels.index("donut")]=set(actionsObjects[labels.index("banana")])
	actionsObjects[labels.index("cake")]=set(actionsObjects[labels.index("banana")])

	actionsObjects[labels.index("cell phone")]={"call","usetelephone"}
	actionsObjects[labels.index("laptop")]={"work","chat","uselaptop"}
	actionsObjects[labels.index("keyboard")]=set(actionsObjects[labels.index("laptop")])
	actionsObjects[labels.index("mouse")]=set(actionsObjects[labels.index("keyboard")])
	actionsObjects[labels.index("tvmonitor")]={"watchtv"}.union(actionsObjects[labels.index("laptop")])
	actionsObjects[labels.index("remote")]={"watchtv"}

	actionsObjects[labels.index("chair")]={"rest","sit","sitdown","getup"}
	actionsObjects[labels.index("diningtable")]={"eat"}
	actionsObjects[labels.index("sofa")]={"rest","sit","sitdown","getup","laydown"}
	actionsObjects[labels.index("bed")]={"rest","sleep","sitdown","getup","laydown"}

	actionsObjects[labels.index("cat")]={"feed","play","carry"}
	actionsObjects[labels.index("bird")]=set(actionsObjects[labels.index("cat")])
	actionsObjects[labels.index("dog")]={"walk the dog"}.union(actionsObjects[labels.index("cat")])

	actionsObjects[labels.index("backpack")]={"fill it", "tak inside object","put inside objects","pick it up"}
	actionsObjects[labels.index("handbag")]=set(actionsObjects[labels.index("backpack")])
	actionsObjects[labels.index("suitcase")]=set(actionsObjects[labels.index("backpack")])

	actionsObjects[labels.index("bottle")]={"drink","pour"}
	actionsObjects[labels.index("wine glass")]=set(actionsObjects[labels.index("bottle")])
	actionsObjects[labels.index("cup")]={"drink","pour"}'''

	#Parche -> FALTA solucionar de forma bonita: Acciones para pruebas con Toyota
	actionsObjects[labels.index("book")]={"readbook"}

	actionsObjects[labels.index("refrigerator")]={"eat","cook"}
	actionsObjects[labels.index("microwave")]={"eat","cook"}
	actionsObjects[labels.index("toaster")]={"eat","cook"}

	actionsObjects[labels.index("fork")]={"eat"}
	actionsObjects[labels.index("knife")]=set(actionsObjects[labels.index("fork")])
	actionsObjects[labels.index("spoon")]=set(actionsObjects[labels.index("fork")])
	actionsObjects[labels.index("bowl")]={"eat","pour"}

	actionsObjects[labels.index("banana")]={"eat"}
	actionsObjects[labels.index("apple")]=set(actionsObjects[labels.index("banana")])
	actionsObjects[labels.index("sandwich")]=set(actionsObjects[labels.index("banana")])
	actionsObjects[labels.index("broccoli")]=set(actionsObjects[labels.index("banana")])
	actionsObjects[labels.index("carrot")]=set(actionsObjects[labels.index("banana")])
	actionsObjects[labels.index("hot dog")]=set(actionsObjects[labels.index("banana")])
	actionsObjects[labels.index("pizza")]=set(actionsObjects[labels.index("banana")])
	actionsObjects[labels.index("donut")]=set(actionsObjects[labels.index("banana")])
	actionsObjects[labels.index("cake")]=set(actionsObjects[labels.index("banana")])

	actionsObjects[labels.index("cell phone")]={"usetelephone"}
	actionsObjects[labels.index("laptop")]={"uselaptop"}
	actionsObjects[labels.index("keyboard")]=set(actionsObjects[labels.index("laptop")])
	actionsObjects[labels.index("mouse")]=set(actionsObjects[labels.index("keyboard")])
	actionsObjects[labels.index("tvmonitor")]={"watchtv"}.union(actionsObjects[labels.index("laptop")])
	actionsObjects[labels.index("remote")]={"watchtv"}

	actionsObjects[labels.index("chair")]={"sitdown","getup"}
	actionsObjects[labels.index("diningtable")]={"eat"}
	actionsObjects[labels.index("sofa")]={"sitdown","getup","laydown"}
	actionsObjects[labels.index("bed")]={"sitdown","getup","laydown"}

	actionsObjects[labels.index("bottle")]={"drink","pour"}
	actionsObjects[labels.index("wine glass")]=set(actionsObjects[labels.index("bottle")])
	actionsObjects[labels.index("cup")]={"drink","pour","maketea","makecaffee"}

	#---------------------CARGAMOS ACCIONES DEL TUHOI DATASET------------------------#
	with open('../../tuhoi_dataset.csv',errors='ignore') as csv_file:
	#with open('../../tuhoi_dataset.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			obj = row[18]
			if obj in labels:
				for act in row[6].split('\n'):
					act=act.lower()
					ind = labels.index(obj)
					if not act in actionsToIgnore and act in actionsInToyota: #Segunda parte solo para las pruebas:
						actionsObjects[ind].add(act)
			
			obj = row[21]
			if obj in labels:
				for act in row[8].split('\n'):
					act=act.lower()
					ind = labels.index(obj)
					if not act in actionsToIgnore and act in actionsInToyota: #Segunda parte solo para las pruebas:
						actionsObjects[ind].add(act)

			line_count+=1
			#Por problemas de codificacion
			if line_count >= 10764:
				break
		print(f'Processed {line_count} lines.')
	#--------------------------------------------------------------------------------#

	#----------------------------BORRAR---------------------------------
	'''for j in range(len(actionsObjects)):
		print('Objeto ' + labels[j])
		for act in actionsObjects[j]:
			if act in actionsInToyota:
				print(act)
		print("")'''

	'''for obj in actionsObjects:
		for act in obj:
			print(act)
		print('\n')'''

	return actionsObjects

# interactionableObjects = defineInteractionableObjects()
actionsObjects = defineActionsObjects()

#Step9: Test yolo --------Prueba1
# load and prepare an image
def load_image_pixels(image, shape):
	# load the image to get its shape
	#image = load_img(filename)
	shapeImg = image.shape
	#width, height = sh[0],sh[1]
	# load the image with the required size
	#image = load_img(filename, target_size=shape)
	image = cv2.resize(image, dsize=shape, interpolation=cv2.INTER_CUBIC)
	# convert to numpy array
	#image = img_to_array(image)
	# scale pixel values to [0, 1]
	image = image.astype('float32')
	image /= 255.0
	# add a dimension so that we have one sample
	image = expand_dims(image, 0)
	return image, shapeImg[0], shapeImg[1]

# define the expected input shape for the model
input_w, input_h = 416, 416

########## PARA VARIOS FRAMES #######################
#Cargamos el video
#vidcap = cv2.VideoCapture('/mnt/md1/datasets/ETRI-Activity/P001-P010/P001/A001_P001_G001_C001.mp4')
#vidcap = cv2.VideoCapture('/mnt/md1/datasets/ETRI-Activity/P001-P010/P001/A001_P001_G002_C004.mp4')
#vidcap = cv2.VideoCapture('/mnt/md1/datasets/toyota_smarthome/rgb/mp4/Drink.Fromcan_p16_r01_v14_c06.mp4')
import glob
namesVideos = glob.glob("/mnt/md1/datasets/toyota_smarthome/rgb/mp4/*.mp4")

aciertos=0
videosAnalizados=0
numTotalPredictedActions=0
aciertosForAct = [0] * len(actionsInToyota)
fallosForAct = [0] * len(actionsInToyota)
for name in namesVideos:
	vidcap = cv2.VideoCapture(name)
	actionReal=name[43:]
	actionReal=actionReal.split('_')[0]
	actionReal=actionReal.split('.')[0]
	actionReal=actionReal.lower()

	if actionReal!='walk' and actionReal!='enter' and actionReal!='leave':

		#vidcap = cv2.VideoCapture('/mnt/md1/datasets/ETRI-Activity/P001-P010/P001/A055_P001_G003_C007.mp4')
		#Leemos frame
		success,imageOrig = vidcap.read()
		count = 0
		totalLabels = list()
		totalIndex = list()
		totalBoxes = list()
		totalScores = list()
		while success:
			#if count%20==0: #El video va a 20 frames por segundo
			if count%1000==0:
				#photo_filename = 'images/frame.jpg'
				#cv2.imwrite(photo_filename, imageOrig)     # save frame as JPEG file
				#imageOrig=Image.fromarray(np.uint8(imageOrig))
				#print(type(imageOrig)) #Borrar
				#Ajustamos los valores de la imagen
				image, image_w, image_h = load_image_pixels(imageOrig, (input_w, input_h))
				#Realizamos la prediccion
				yhat = yolov3.predict(image)
				boxes = list()
				########################### OPTIMIZAR ###########################################
				for i in range(len(yhat)):
					# decode the output of the network
					boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
				# correct the sizes of the bounding boxes for the shape of the image
				correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
				# suppress non-maximal boxes
				#do_nms(boxes, 0.5)
				#################################################################################
				########################### OPTIMIZAR ###########################################
				# get the details of the detected objects
				v_labels, v_index = get_boxes_light(boxes, labels, class_threshold)	#OPTIMIZAR
				for i in range(len(v_labels)):
					if not v_index[i] in totalIndex:
						totalIndex.append(v_index[i])
						totalLabels.append(v_labels[i])
						#totalBoxes.append(v_boxes[i])
						#totalScores.append(v_scores[i])
				#################################################################################
			#Leemos siguiente frame
			success,imageOrig = vidcap.read()
			#print(f'Readed frame: {count}', success) #BORRAR
			count += 1

		possibleActions=set()

		for i in range(len(totalLabels)):
			#print('Objeto: ' + labels[totalIndex[i]])	#Borrar
			#print('Acciones:  \n')	#Borrar
			for act in actionsObjects[totalIndex[i]]:
				#print(act)	#Borrar
				possibleActions.add(act)

		videosAnalizados+=1
		numTotalPredictedActions+=len(possibleActions)

		if actionReal in possibleActions:
			aciertosForAct[actionsInToyota.index(actionReal)]+=1
			aciertos+=1
		else:
			fallosForAct[actionsInToyota.index(actionReal)]+=1
			print('Esperaba: ' + actionReal)
			print('Archivo: ' + name)
			'''print('Acciones posibles:')
			for actAuxToPrint in possibleActions:
				print(actAuxToPrint)'''
		
		print('Aciertos: ' + str(aciertos) + '/' + str(videosAnalizados))
		print('Acciones por video: ' + str(numTotalPredictedActions/videosAnalizados))
		print('Aciertos de la accion ' + actionReal + ': ' + str(aciertosForAct[actionsInToyota.index(actionReal)]) + '/'
		+ str(aciertosForAct[actionsInToyota.index(actionReal)] + fallosForAct[actionsInToyota.index(actionReal)]))
		#print([a.shape for a in yhat])


		# summarize what we found
		'''for i in range(len(totalLabels)):
			#print(v_labels[i], v_scores[i])
			print(totalLabels[i])
			print("Possible actions:")
			for act in actionsObjects[totalIndex[i]]:
				print(act)
			print()
		# draw what we found
		draw_boxes(photo_filename, totalBoxes, totalLabels, totalScores)'''



######################################################

######### PARA UNA UNICA IMAGEN ################
# define our new photo
# photo_filename = 'images/frame.jpg'
# load and prepare image
#image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))

# make prediction
#yhat = yolov3.predict(image)
# summarize the shape of the list of arrays
# print([a.shape for a in yhat])

# boxes = list()
# for i in range(len(yhat)):
# 	# decode the output of the network
# 	boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
# # correct the sizes of the bounding boxes for the shape of the image
# correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
# # suppress non-maximal boxes
# do_nms(boxes, 0.5)

# # get the details of the detected objects
# v_boxes, v_labels, v_scores, v_index = get_boxes(boxes, labels, class_threshold)
# # summarize what we found
# for i in range(len(v_boxes)):
# 	print(v_labels[i], v_scores[i])
# 	print("Possible actions:")
# 	for act in actionsObjects[v_index[i]]:
# 		print(act)
# 	print()
# # draw what we found
# draw_boxes(photo_filename, v_boxes, v_labels, v_scores)