import cv2
import numpy as np
import glob
# NOTE:
# -This image classifier works by combining the ORB and SIFT algorithms.
# -The ORB descriptors are 32-length.
# - The SIFT descriptors are 128-length and thus need to be reshaped.

#define paths to images and store them
path_query = r'/Users/avimalhotra/Desktop/CS4186 assignment1/datasets_4186/cropped_query_4186'
path_gallery = r'/Users/avimalhotra/Desktop/CS4186 assignment1/datasets_4186/gallery_4186'

name_query = glob.glob(path_query + '/*.jpg')
num_query = len(name_query)

name_gallery = glob.glob(path_gallery + '/*.jpg')
num_gallery = len(name_gallery)

# initiate ORB & SIFT detectors
orb = cv2.ORB_create(nfeatures=1000)
sift = cv2.SIFT_create()
all_records = np.zeros((num_query, len(name_gallery)))

# Flann algorithm parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

# open file for storing the results
f = open(r'./rank_list_ORB_SIFT.txt', 'w')

# iterate over all the images in the query folder
for query in name_query:
    # compute and reshape SIFT descriptors
    cur_img = cv2.imread(query, cv2.IMREAD_GRAYSCALE)
    sift_kp_img, sift_desc_img = sift.detectAndCompute(cur_img, None)
    new_sift_desc = sift_desc_img.reshape((int(128/32) * sift_desc_img.shape[0], 32))

    # compute ORB descriptors and concatenate with SIFT descriptors
    orb_kp_img, orb_desc_img = orb.detectAndCompute(cur_img, None)
    all_desc_img = np.concatenate((new_sift_desc, orb_desc_img), axis=0)

    dist_record = []
    # iterate over all the images in the gallery folder and find the matches
    for gallery in name_gallery:
        # compute and reshape SIFT descriptors
        gal_img = cv2.imread(gallery, cv2.IMREAD_GRAYSCALE)
        sift_kp_gal, sift_desc_gal = sift.detectAndCompute(gal_img, None)
        new_sift_desc_gal = sift_desc_gal.reshape((int(128 / 32) * sift_desc_gal.shape[0], 32))

        # compute ORB descriptors and concatenate with SIFT descriptors
        orb_kp_gal, orb_desc_gal = orb.detectAndCompute(gal_img, None)
        all_desc_gal = np.concatenate((new_sift_desc_gal, orb_desc_gal), axis=0)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(all_desc_img, all_desc_gal, k=2)
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
