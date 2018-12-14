from google_images_download import google_images_download
from pathlib import Path
import pandas as pd
import numpy as np
from topic_modeler import load_data
from time import sleep


class GoogleImageDownloader(object):
	"""
	Requires:
	1. google_images_download @ https://github.com/hardikvasa/google-images-download
	2. Chromedriver
	3. Selenium


	GoogleImageDownloader has two methods for downloading images:
		1. From a list of words 2. For a specific keyword.

	Example:
	GoogleImageDownloader().download_images_from_list(wordlist, # pictures, download_diretory)

	GoogleImageDownloader().download_images_keyword(word, # pictures, download_directory)

	"""

	def __init__(self):
		pass

	def download_images_from_list(self, wordlist, limit, image_directory_name):
		"""
		Inputs: wordlist (list of strings)
				limit (integer)
				image_directory_name (directory path as string)

		Outputs:
				A directory with images downloaded from Google
		"""

		for word in wordlist:
			self.download_images_keyword(word, limit, image_directory_name)


	def download_images_keyword(self, keyword, limit, image_directory_name):
		"""
		Inputs: keyword (string)
				limit (integer)
				image_directory_name (directory path as string)

		Outputs: A directory with images downloaded from Google
		"""

		cwd = str(Path.cwd())
		arguments = {
					"keywords": keyword, "limit" : limit, "output_directory" : f'{cwd}/',
					"image_directory" : image_directory_name, "print_urls" : True,
					"chromedriver" : "/usr/local/bin/chromedriver"
					}
		response = google_images_download.googleimagesdownload()
		paths = response.download(arguments)
		print(paths)



if __name__ == '__main__':
	gid = GoogleImageDownloader()
	df = load_data()
	df['images'] = 'no image'


	# for beer in df['beer_name']:
	# 	data = gid.download_images_keyword(beer + ' ' + 'in glass',3, 'data/train')
	# 	import pdb; pdb.set_trace()
	# 	break

	# gid.download_images_keyword("stout beers in glass", 100, 'data/train')
	# gid.download_images_keyword("hazy beers in glass", 100, 'data/train')
	# gid.download_images_keyword("pilsner beers in glass", 100, 'data/train')
	# gid.download_images_keyword("ipa beers in glass", 100, 'data/train')
