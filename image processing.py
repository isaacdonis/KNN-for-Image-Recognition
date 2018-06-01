import os
import numpy as np
import datetime
from scipy.spatial.distance import euclidean
from PIL import Image
from IPython.display import display

# get the image directories
image_dir = r'C:\Users\IsaacDonis\Desktop\Computer Vision Materials\data\flowers-recognition\flowers\flowers'
print("The code is starting at " + str(datetime.datetime.now()))
start = datetime.datetime.now()

class KNN_Classifier(object):

	def __init__(self, labels, samples):
		""" The first part of the KNN Classifier is simply to memorize all of the data points.
			This is the training part

			labels (array): an array showing which labels are associated with the various images
			samples (array of Image objects): the training data, a set of images
			k: the number of classes
		"""

		self.labels = labels 
		self.samples = samples

	def classify(self, point, k):
		"""
			Here, we define the maximum number of training images that we want to find similar
			to the image that we are trying to classify with k=3. 

			Returns - a dictionary with the number of votes that a particular label gets

			point (Image object): The Image we are trying to classify
			k (int): The maximum number of images in the traing set we want to use to vote for which class point is
		"""

		# compute the L2 (Euclidian) distance of the given images to all other images in the training set
		distance = np.array([self.L2_distance(point, the_sample) for the_sample in self.samples])
		
		# get rid of when you compare the point to itself 
		distance = distance[distance != 0]

		# sort the distances 
		index = distance.argsort()

		# use a dictionary to store the k nearest neighbors
		votes = {}

		for i in range(k):

			label = self.labels[index[i]]
			votes.setdefault(label, 0)
			votes[label] +=1

		return votes

	def L2_distance(self, point_one, point_two):

		"""
			Compute the L2 distance between a single image and all the other images in the dataset 

			Returns - the L2 distance between two images
			
			point_one: the image you are trying to classify
			point_two: another image from the dataset that you are trying to compare 

		"""
		return euclidean(np.array(point_one).ravel(), np.array(point_two).ravel())	
	
list_of_flowers = os.listdir(image_dir)

# create a new dictionary where the key will be the flower and the value is a list of image directories
flowers = {}
counter = 0 

for item in list_of_flowers:
	
	list_of_images =  os.listdir(os.path.join(image_dir, item))
	abs_path_of_images = [image_dir + '\\' + os.path.join(item, the_image) for the_image in list_of_images]
	flowers[item] = [Image.open(the_image).resize((256, 256), Image.ANTIALIAS) for the_image in abs_path_of_images]
	print(counter + 1)

# create a dictionary mapping the different flowers to a label
print("Done resizing, etc")
flower_mapping = dict(zip(np.arange(len(flowers.keys())), sorted(flowers.keys())))

# get the labels and samples into a list format 
labels = []
for flower_label in flower_mapping.keys():

	labels.extend(str(flower_label) * len(flowers[flower_mapping[flower_label]]))

# turn the label back into a list of ints
labels = list(map(int, labels))

samples = []
for key in flower_mapping.values():

	samples.extend(flowers[key])

# create an instance of the model
model = KNN_Classifier(labels=labels, samples=samples)

# define k
k = len(set(labels))

num_correct = 0
print("Predicting the labels...")
for i in range(len(samples)):

	predicted_label = max(model.classify(point=model.samples[i], k=k))

	if predicted_label == labels[i]:
		num_correct += 1 

print(num_correct / len(samples))


print("The code took " + str(start-datetime.datetime.now() + " time to run."))