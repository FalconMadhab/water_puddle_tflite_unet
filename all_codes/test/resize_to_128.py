import cv2
import os
#Size of test input images should be 128
#Please run the code to resize the test images
#before testing 

images_folder_path = "/U-Net/code/test/test_2D_satellite_images/"
save_resized_image = "/U-Net/code/test/test_128/"
for filename in os.listdir(images_folder_path):
    image = cv2.imread(images_folder_path+filename)
    resized_image = cv2.resize(image, (128, 128)) 
    image_resized = save_resized_image + filename
    print("resizing")
    cv2.imwrite(image_resized,resized_image)