import glob
import numpy as np
from PIL import Image

# Note:This script simply extract the bounding box information and stores the corresponding images in a new folder.

# Defined path to the images
path_query = r'/Users/avimalhotra/Desktop/CS4186 assignment1/datasets_4186/query_4186'
path_query_text = r'/Users/avimalhotra/Desktop/CS4186 assignment1/datasets_4186/query_txt_4186'
cropped_query = r'/Users/avimalhotra/Desktop/CS4186 assignment1/datasets_4186/cropped_query_4186'

# Imports and sorts the images
name_query = glob.glob(path_query + '/*.jpg')
text_query = glob.glob(path_query_text + '/*.txt')
num_query = len(name_query)

name_query.sort()
text_query.sort()


# Crops an image as per the bounding box
def query_crop(query_path, txt_path):
    img = Image.open(query_path)
    txt = np.loadtxt(txt_path)
    img = img.crop((int(txt[0]), int(txt[1]), int(txt[0] + txt[2]), int(txt[1] + txt[3])))
    img.save(cropped_query + '/' + query_path.split('/')[-1])


def main():
    for i in range(num_query):
        query_crop(name_query[i], text_query[i])


if __name__ == '__main__':
    main()
