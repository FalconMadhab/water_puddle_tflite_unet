import cv2
import os
import ast
import shutil  
import numpy as np 

images_folder_path = "/data/here/images/"
Label_folder_path = "/here/data/here/labels_stitched_1280x1280/poly_xy_footprints/"
save_mask_folder_path = "/here/data/here/labels_stitched_1280x1280/masks/"
training_image_folder = "/here/data/Train/"
mask_size = 1280
count  = 0

for filename in os.listdir(images_folder_path):
	
	if filename.startswith("._"):
		print("XXXXXX",filename)
		continue	
	else:
		count =count + 1
		Image_Id =  filename.split('.')[0]
		folder_per_Image_Id = training_image_folder + Image_Id +"/"
		images_per_Image_Id = folder_per_Image_Id + "images/"
		print("images_per_Image_Id",images_per_Image_Id)
		mask_per_Image_Id = folder_per_Image_Id + "masks/"
		print("mask_per_Image_Id",mask_per_Image_Id)

		if not os.path.exists(folder_per_Image_Id):
			os.makedirs(folder_per_Image_Id)

		if not os.path.exists(images_per_Image_Id):
			os.makedirs(images_per_Image_Id)

		if not os.path.exists(mask_per_Image_Id):
			os.makedirs(mask_per_Image_Id)

		source_image_path = images_folder_path +  filename
		destination_image_path = images_per_Image_Id + filename
		shutil.copyfile(source_image_path,destination_image_path)


		f = open(Label_folder_path+filename.split('.')[0]+".txt")

		print("filename------------>",count,filename)
		
		save_mask_filename = 0
		for each_contour in f:
			mask = np.zeros( (mask_size,mask_size) ) 
			save_mask_filename = save_mask_filename + 1
			str_save_mask_filename = str(save_mask_filename)
			arr = ast.literal_eval(each_contour)
			arr = np.array(arr)
			cv2.fillPoly(mask, [arr], color=(255))
			save_mask_filename_full = mask_per_Image_Id + str_save_mask_filename + filename
			cv2.imwrite(save_mask_filename_full,mask)

