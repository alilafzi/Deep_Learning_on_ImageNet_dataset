''' Part of this code is borrowed from the given reference in the homework handout, i.e. github.com/johancc/ImageNetDownloader  '''
#import torchvision
#import torch.utils.data
import glob
import os
import numpy
import PIL
import argparse 
import requests
import logging
import json

parser = argparse.ArgumentParser ( description = 'HW02 Task1')
parser.add_argument ( '--subclass_list' , nargs = '*' , type = str
, required = True )
parser.add_argument ( '--images_per_subclass' , type = int ,
required = True) 
parser.add_argument ( '--data_root' , type = str , required =
True )
parser.add_argument ( '--main_class' , type = str , required =
True )
parser.add_argument ( '--imagenet_info_json' , type = str ,
required = True )
args , args_other = parser.parse_known_args ()

os.chdir(args.imagenet_info_json)
with open('imagenet_class_info.json') as f:
	jsondata = json.load(f)
#print (jsondata)

from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL
from PIL import Image

def get_image(img_url, class_folder):
	if len(img_url) <= 1:
		return 'useless'

	try:
		img_resp = requests.get(img_url, timeout = 1)
	except ConnectionError:
		return 'useless'
	except ReadTimeout:
		return 'useless'
	except TooManyRedirects:
		return 'useless'
	except MissingSchema:
		return 'useless' 
	except InvalidURL:
		return 'useless'

	if not 'content-type' in img_resp.headers:
		return 'useless'

	if not 'image' in img_resp.headers['content-type']:
		return 'useless'

	if (len(img_resp.content) < 1000):
		return 'useless'

	img_name = img_url.split('/')[-1]
	img_name = img_name.split("?")[0]

	if (len(img_name) <= 1):
		return 'useless'

	if not 'flickr' in img_url :
		return 'useless'

	img_file_path = os.path.join(class_folder, img_name)

	with open(img_file_path, 'wb') as img_f:
		img_f.write(img_resp.content)

        
	# Resize image to 64x64
	im = Image . open ( img_file_path )

	if im.mode != "RGB":
		im = im.convert ( mode = "RGB" )

	im_resized = im . resize (( 64 , 64 ) , Image . BOX )
	# Overwrite original image with downsampled image
	im_resized . save ( img_file_path )


IMAGENET_API_WNID_TO_URLS = lambda wnid: f'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={wnid}'

imagenet_images_folder = os.path.join(args.data_root)
if not os.path.isdir(imagenet_images_folder):
	os.mkdir(imagenet_images_folder)
#print (imagenet_images_folder)
os.chdir(imagenet_images_folder)
os.mkdir(args.main_class)
os.chdir(args.main_class)
imagenet_images_folder = os.getcwd()

#classes_to_scrape = ["n02123597"]
classes_to_scrape = []
for desired in args.subclass_list: 
	for key in jsondata.keys():
		if jsondata[key]["class_name"] == desired:
			classes_to_scrape.append(key)
#print (classes_to_scrape)

def run(classes_to_scrape: list = classes_to_scrape):
	
	for class_wnid in classes_to_scrape:
		counter = 0
		class_name = jsondata[class_wnid]["class_name"]
		
		url_urls = IMAGENET_API_WNID_TO_URLS(class_wnid)

		resp = requests.get(url_urls)

		class_folder = os.path.join(imagenet_images_folder, class_name)
		if not os.path.exists(class_folder):
			os.mkdir(class_folder)

		urls = [url.decode('utf-8') for url in resp.content.splitlines()]
		
		for url in  urls:
			counter += 1
			if counter > args.images_per_subclass:
				break
					
			filenames_before_download = os.listdir(class_folder)
			
			get_image(url, class_folder)
			
			##check if an image is already downloaded
			filenames_after_download = os.listdir(class_folder)
			
			if get_image(url, class_folder) == 'useless':
				counter -= 1
			else:
				if len(filenames_before_download) == len(filenames_after_download):
					counter -= 1

if __name__ == "__main__":
	run()
