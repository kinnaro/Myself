import cv2
import numpy as np



def read_image(tuple_set):
	file_path = tuple_set[0]
	label = tuple_set[1]
	img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
	# 你这里的参数，可以是彩色或者灰度(GRAYSCALE)
	return img, label


###########
# 预处理####
###########
CHANNELS = 1
# 代表1个颜色频道


def prep_data(images):
	no_images = len(images)
	data = np.ndarray((no_images, CHANNELS, 28, 28), dtype=np.uint8)
	labels = []

	for i, image_file in enumerate(images):
		image, label = read_image(image_file)
		data[i] = image.T
		labels.append(label)
		return data, labels
