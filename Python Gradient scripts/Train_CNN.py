# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, shutil
import sys
sys.path.append(r"C:\Users\labuser\Desktop\GRADIENT_CLEX")
import numpy as np
import tensorflow as tf


from Img_loader_2_classes import bead_loader

from bead_model_120_x_120_10x import bead_model_fn

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


image_size_width=478
image_size_height=478
#main_path=r'C:\Users\labuser\Desktop\'
main_path=r'Image Collection'

model_dir=r'C:\Users\labuser\Desktop\GRADIENT_CLEX\new-model'

def main(unused_argv):
           
     # bead_classifier = tf.estimator.Estimator(model_fn=bead_model_fn, model_dir=model_dir)
      bead_classifier = tf.compat.v1.estimator.Estimator(model_fn=bead_model_fn, model_dir=model_dir)
     
      #Train the model
      #train_input_fn = tf.estimator.inputs.numpy_input_fn(
      train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
          x={"x": training_imgs},
          y=training_labels,

          batch_size=1,
          num_epochs=None,
          shuffle=True) 
      bead_classifier.train(
          input_fn=train_input_fn,
          steps = 8000)# calculate steps for at least 3 epochs
          #hooks=[logging_hook])
    
      # Evaluate the model and print results
      
      eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
          x={"x": validation_imgs},
          y=validation_labels,
          num_epochs=1,
          shuffle=False)
      
      eval_results = bead_classifier.evaluate(input_fn=eval_input_fn)
      print(eval_results)
            
      #tf.train.export_meta_graph(filename=model_dir+r'\my-model.meta')
      #print("Model saved in file")
 
if __name__ == "__main__":
  # Load training and eval data
  #res = bead_loader(image_size_width,image_size_height,main_path)
  res = bead_loader(main_path)

  training_imgs = np.array(res[0], dtype = np.float32)
  training_labels = np.asarray(res[1], dtype=np.int32)
  validation_imgs = np.array(res[2], dtype = np.float32)
  validation_labels = np.asarray(res[3], dtype=np.int32)
  print(training_labels.shape)    
  tf.compat.v1.app.run()