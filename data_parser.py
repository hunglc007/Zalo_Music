import os
import numpy as np
from PIL import Image
from config import slices_path,slices_path_stride,slices_val

def read_file_list(filelist):

	pfile = open(filelist)
	filenames = pfile.readlines()
	pfile.close()

	filenames = [f.strip() for f in filenames]

	return filenames

def split_pair_names(filenames, base_dir):

	filenames = [c.split(' ') for c in filenames]
	filenames = [(os.path.join(base_dir, c[0]), os.path.join(base_dir, c[1])) for c in filenames]

	return filenames


class DataParser():

	def __init__(self, batch_size_train):
		countfile = 0
		self.samples = []
		genresID = {'CaiLuong': 0, 'CachMang': 1, 'DanCa': 2, 'Dance': 3, 'KhongLoi': 4, 'ThieuNhi': 5, 'Trinh': 6,
					'TruTinh': 7, 'Rap': 8, 'Rock': 9}
		rootdirectory = slices_path_stride
		listdir = [d for d in os.listdir(rootdirectory) if os.path.isdir(os.path.join(rootdirectory, d))]
		limit = 29000
		for idx, dir in enumerate(listdir):
			listfile = os.listdir(rootdirectory + '/' + dir)
			for count, file in enumerate(listfile):
				if count >= limit and rootdirectory == slices_path: break
				genre = file.split("_")[0]
				filepath = rootdirectory + '/' + dir + '/' + file
				self.samples.append([filepath,genresID[genre]])
				print(filepath + '   :    ' + str(count))

		countfile = len(self.samples)
		np.random.shuffle(self.samples)
		rootdirectory = slices_val
		listdir = [d for d in os.listdir(rootdirectory) if os.path.isdir(os.path.join(rootdirectory, d))]
		for idx, dir in enumerate(listdir):
			for count, file in enumerate(os.listdir(rootdirectory + '/' + dir)):
				genre = file.split("_")[0]
				filepath = rootdirectory + '/' + dir + '/' + file
				self.samples.append([filepath, genresID[genre]])
				print(filepath + '   :    ' + str(count))

		self.n_samples = len(self.samples)
		self.all_ids = list(range(self.n_samples))
		self.training_ids = self.all_ids[:countfile]
		self.validation_ids = self.all_ids[countfile:]

		self.batch_size_train = batch_size_train
		self.steps_per_epoch = len(self.training_ids)/batch_size_train

		self.validation_steps = len(self.validation_ids)/(batch_size_train)

		self.image_width = 128
		self.image_height = 128
		self.num_channel = 1
		self.target_regression = True
	def load_config(self):
		return self.image_height, self.image_width, self.num_channel, len(self.training_ids), len(self.validation_ids)
	def get_training_batch(self):

		batch_ids = np.random.choice(self.training_ids, self.batch_size_train)

		return self.get_batch(batch_ids)

	def get_validation_batch(self):

		batch_ids = np.random.choice(self.validation_ids, self.batch_size_train*2)

		return self.get_batch(batch_ids)

	def get_batch(self, batch):

		filenames = []
		images = []
		edgemaps = []

		for idx, b in enumerate(batch):
			im = Image.open(self.samples[b][0])
			im = im.resize((self.image_width, self.image_height))
			im = np.asarray(im, dtype=np.uint8).reshape(128, 128, 1)
			im = im / 255.
			em = self.samples[b][1]
			label = [1. if em == g else 0. for g in range(10)]
			images.append(im)
			edgemaps.append(label)
			filenames.append(self.samples[b])
		images   = np.asarray(images)
		edgemaps = np.asarray(edgemaps)

		return images, edgemaps



