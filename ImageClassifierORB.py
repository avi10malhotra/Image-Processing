import cv2
import numpy as np
import glob

# define paths to images and store them
path_query = r'/Users/avimalhotra/Desktop/CS4186 assignment1/datasets_4186/cropped_query_4186'
path_gallery = r'/Users/avimalhotra/Desktop/CS4186 assignment1/datasets_4186/gallery_4186'

name_query = glob.glob(path_query + '/*.jpg')
num_query = len(name_query)

name_gallery = glob.glob(path_gallery + '/*.jpg')
num_gallery = len(name_gallery)

# create ORB object
orb = cv2.ORB_create(nfeatures=1000)
all_records = np.zeros((num_query, len(name_gallery)))

# open file for storing the results
f = open(r'./rank_list_ORB.txt', 'w')

# iterate over all the images in the query folder
for query in name_query:
    # read the images in grayscale and compute the keypoints + descriptors
    cur_img = cv2.imread(query, cv2.IMREAD_GRAYSCALE)
    kp_img, desc_img = orb.detectAndCompute(cur_img, None)
    dist_record = []
    # iterate over all the images in the gallery folder and find the matches
    for gallery in name_gallery:
        gal_img = cv2.imread(gallery, cv2.IMREAD_GRAYSCALE)
        kp_gal, desc_gal = orb.detectAndCompute(gal_img, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_img, desc_gal, k=2)
        good_matches = 0
        # store the good matches that fulfill the criterion
        for x, y in matches:
            if x.distance < 0.75 * y.distance:
                good_matches += 1
        dist_record.append((gallery.split('/')[-1].strip('.jpg'), good_matches))
    # sort the gallery images according to the number of "good" matches
    dist_record.sort(key=lambda x: x[1], reverse=True)

    f.write('Q' + query.split('/')[-1].strip('.jpg') + ': ')
    for j in range(num_gallery):
        f.write(str(dist_record[j][0]) + ' ')
    f.write('\n')

f.close()
