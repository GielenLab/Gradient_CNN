# -*- coding: utf-8 -*-

import os
import numpy as np
import skimage.io as skio
from pathlib import Path

def bead_loader(folder):
    

    bead_path = Path(os.getcwd())
    
    bead_path = bead_path / folder
    
    training_path = bead_path /'Training'
    validation_path = bead_path /'Validation'
    testing_path = bead_path /'Testing'
    
    
    testing_imgs = []
    training_imgs = []
    validation_imgs = []
    
    testing_labels = []
    training_labels = []
    validation_labels = []
    
    imgs = []
    label_defs = [0, 1]

    
    #--------------------TESTING IMAGES---------------------------
    foldrs = os.listdir(testing_path)
    for i in range(2):
        files = []    
        loop_path = testing_path / foldrs[i]
    
        for f in os.listdir(loop_path):
            filename = os.fsdecode(f)
            if  filename.endswith(".png"): 
                files.append(filename)
        
        imgs = np.array([np.array(np.float16(skio.imread(loop_path / fname))) for fname in files])    
        testing_imgs.append(imgs)
        
        labs = np.ones(testing_imgs[i].shape[0])*label_defs[i]
        testing_labels.append(labs)
        
    testing_imgs = np.array(np.concatenate((testing_imgs[0], testing_imgs[1]), axis = 0), dtype = np.float32)
    for counter, f in enumerate(testing_imgs):
        
        f = f-np.mean(f)
        f = f/np.std(f)
        testing_imgs[counter] = f
    testing_imgs = np.reshape(testing_imgs, (testing_imgs.shape[0], testing_imgs.shape[1]*testing_imgs.shape[2]))
    
    testing_labels = np.asarray(np.concatenate((testing_labels[0], testing_labels[1]), axis = 0), dtype = np.int)
    print("Testing images loaded...")
    #--------------------TRAINING IMAGES---------------------------
    foldrs = os.listdir(training_path)
    
    for i in range(2):
        files = []    
        loop_path = training_path / foldrs[i]

        for f in os.scandir(loop_path): #inside each subfolder
                filename = os.fsdecode(f)
                if  filename.endswith(".png"): 
                    files.append(filename)
        print("{} images for class {} loading...".format(len(files),i))
        
        imgs = np.array([np.array(np.float16(skio.imread(fname))) for fname in files])    
        training_imgs.append(imgs)
        
        labs = np.ones(training_imgs[i].shape[0])*label_defs[i]
        training_labels.append(labs)
    
    #training_imgs = np.array(np.concatenate((training_imgs[0], training_imgs[1],training_imgs[2],training_imgs[3], training_imgs[4],training_imgs[5],training_imgs[6], training_imgs[7],training_imgs[8]), axis = 0), dtype = np.float32)
    training_imgs = np.array(np.concatenate((training_imgs[0], training_imgs[1]), axis = 0), dtype = np.float32)
    for counter, f in enumerate(training_imgs):
        f = f - np.mean(f)
        f = f/np.std(f)
        training_imgs[counter] = f
    training_imgs = np.reshape(training_imgs, (training_imgs.shape[0], training_imgs.shape[1]*training_imgs.shape[2]))
    
   # training_labels = np.asarray(np.concatenate((training_labels[0], training_labels[1], training_labels[2],training_labels[3], training_labels[4], training_labels[5],training_labels[6], training_labels[7], training_labels[8]), axis = 0), dtype = np.int)
    training_labels = np.asarray(np.concatenate((training_labels[0], training_labels[1]), axis = 0), dtype = np.int)
   
 #--------------------VALIDATION IMAGES---------------------------
    foldrs = os.listdir(validation_path)
    for i in range(2):
        files = []    
        loop_path = validation_path / foldrs[i]
    
        for f in os.listdir(loop_path):
            filename = os.fsdecode(f)
            if  filename.endswith(".png"): 
                files.append(filename)
        
        imgs = np.array([np.array(np.float16(skio.imread(loop_path / fname))) for fname in files])    
        validation_imgs.append(imgs)
        
        labs = np.ones(validation_imgs[i].shape[0])*label_defs[i]
        validation_labels.append(labs)

    #validation_imgs = np.array(np.concatenate((validation_imgs[0], validation_imgs[1], validation_imgs[2],validation_imgs[3], validation_imgs[4], validation_imgs[5],validation_imgs[6], validation_imgs[7], validation_imgs[8]), axis = 0), dtype = np.float32)
    validation_imgs = np.array(np.concatenate((validation_imgs[0], validation_imgs[1]), axis = 0), dtype = np.float32)
    for counter, f in enumerate(validation_imgs):
        f = f - np.mean(f)
        f = f/np.std(f)
        validation_imgs[counter] = f 
    validation_imgs = np.reshape(validation_imgs, (validation_imgs.shape[0], validation_imgs.shape[1]*validation_imgs.shape[2]))
    

    
    #validation_labels = np.asarray(np.concatenate((validation_labels[0], validation_labels[1], validation_labels[2],validation_labels[3], validation_labels[4], validation_labels[5],validation_labels[6], validation_labels[7], validation_labels[8]), axis = 0), dtype = np.int)
    validation_labels = np.asarray(np.concatenate((validation_labels[0], validation_labels[1]), axis = 0), dtype = np.int)
    return training_imgs, training_labels, validation_imgs, validation_labels, testing_imgs, testing_labels
