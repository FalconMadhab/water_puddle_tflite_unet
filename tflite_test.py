import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

image = Image.open("test_128/img_000000161.png")
base_width  = image.size[0]
base_height = image.size[1]

image = image.resize((128, 128), Image.ANTIALIAS)

# Delete alpha channel
print("image.mode ==", image.mode)
if image.mode == "RGBA":
   image = image.convert("RGB")

# Normalization
image = np.asarray(image)
# prepimg = image / 255.0
prepimg = image[np.newaxis, :, :, :]

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model/converted_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
# print(input_details)
output_details = interpreter.get_output_details()
# print(output_details)
input_shape = input_details[0]['shape']
interpreter.set_tensor(input_details[0]['index'], np.array(prepimg, dtype=np.float32))
t1 = time.time()
interpreter.invoke()
print("elapsedtime =", time.time() - t1)
outputs = interpreter.get_tensor(output_details[0]['index'])
# print('outputs:',outputs)
# print(outputs)
output = outputs[0]
preds_test = (output > 0.5).astype(np.uint8)

# To know the counts of 1's and 0's in mask
# unique, counts = np.unique(preds_test, return_counts=True)
# print(dict(zip(unique, counts)))

mask=preds_test
for i in range(mask.shape[0]):
   for j in range(mask.shape[1]):
      if mask[i][j] == 1:
         mask[i][j] = 255
      else:
         mask[i][j] = 0


cv2.imwrite("mask_tflite.png",mask)
