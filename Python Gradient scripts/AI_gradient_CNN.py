# -*- coding: utf-8 -*-

"""Concentration gradient generator: adjust flow rates of up to three syringe pumps in real-time using evaluation of target CNN classes 
"""

from pyqmix import QmixBus, QmixPump, config
import time
import sys
import os.path
from pymba import Vimba
import numpy as np
import tensorflow as tf
import imageio
import csv

sys.path.append(r"D:\QmixSDK\lib\python") #path for all python libraries
from qmixsdk import qmixbus
from qmixsdk import qmixpump
from pumps_function import * #imports all functions from pumps_function


# Call saved model
model_dir=r'D:\new-model' #change folder pointing to trained CNN model

tf.reset_default_graph()

# Specify location of the cropped region at cross-junction
HEIGHT = 120
WIDTH = 120
OffsetY=0
OffsetX=300

Gain=2
Exposure_Time=45 #in microseconds

# Load the trained model and metadata
new_saver= tf.train.import_meta_graph(model_dir + r'\\model.ckpt-3000.meta') 

sess = tf.Session()   

new_saver.restore(sess, tf.train.latest_checkpoint(model_dir)) #

graph = tf.get_default_graph()

# This is the output tensor of the trained model
softmax_tensor = sess.graph.get_tensor_by_name('softmax_tensor:0') 

#assign the 3 pumps
global pump1
global pump2
global pump3
 
global bus

# Qmix device configuration address
config_name2=r"C:\Users\QmixElements\Projects\default_project\Configurations\3syringes"

dia1=4.78 # syringe ID diametter in mm, 4.78mm for 1mL BD plastic 
dia2=4.78
dia3=4.78

# Qmix device configuration.
 # Initialize the connection to the pump system
bus = qmixbus.Bus
bus.open(config_name2,0)
print("Starting bus communication...")
bus.start()
pump=initialize_pump(config_name2)

#assign the 3 pumps a pointer
pump1=pump[1] 
pump2=pump[2] 
pump3=pump[0] 

syringe_define(pump1,pump2,pump3,dia1,dia2,dia3) #set syringe diameters and flow rates to ul/min

print("Initialization done")

pump1.generate_flow(30) #generate flow for oil phase
pump2.generate_flow(8) #generate flow for 0.7% alginate
pump3.generate_flow(3) #generate flow for 2.8% alginate
print("now establishing initial volume fraction...")
time.sleep(8) # delay to establish steady-state flow

# Load camera settings
with Vimba() as vimba:
    # get system object
    system = vimba.getSystem()
    
    # list available cameras (after enabling discovery for GigE cameras)
    if system.GeVTLIsPresent:                                                      
        system.runFeatureCommand("GeVDiscoveryAllOnce")
        time.sleep(0.2)
        cameraIds = vimba.getCameraIds()
        for cameraId in cameraIds:
            print('Camera ID:', cameraId)
    
    # get and open a camera
    camera0 = vimba.getCamera(cameraIds[0])
    camera0.openCamera()
           
    camera0.Height = HEIGHT
    camera0.Width = WIDTH
    camera0.OffsetY = OffsetY
    camera0.OffsetX = OffsetX
        
    camera0.Gain=Gain
    camera0.ExposureTime = Exposure_Time
        
    #Set camera in external trigger mode
    camera0.TriggerMode = 'On'
    camera0.TriggerActivation = 'RisingEdge'
    camera0.PixelFormat = 'Mono8'
    camera0.AcquisitionMode = 'Continuous'
    camera0.ExposureMode = 'Timed'
    camera0.LineSelector = 'Line0'
	
    #set camera registers for correct trigger       
    camera0.writeRegister('0xF0F00614', '00000000')#set trigger pins cf manual
    camera0.writeRegister('0xF0F00830', '82000000')#set trigger pins
    camera0.writeRegister('0xF0F0061C', '80000001')#set trigger pins
    
 
    # create a new frame for the camera
    frame0 = camera0.getFrame()
    
    # announce frame
    frame0.announceFrame()
    # capture camera images
    camera0.startCapture()
    camera0.runFeatureCommand('AcquisitionStart')
    frame0.queueFrameCapture()
       
      
    j=0
    target_class=0 # initial target class= high volume fraction
    target_thresh=0.95 #probability threshold to switch between volume fractions
    
    #initialize average predictions arrays
    pred_result_avg=0 
    pred_probs_avg=0
    pred_result_avg=[]
    pred_probs_avg=[]
    
    time_array=[]

    #stores 50 consecutive values for predictions of classes and confidence
    cnn_result_50=[0]*50 
    cnn_proba_50=[0]*50
    
    step=-1 #step increase for flows in ul/min
    oil_FR=30 # constant oil flow rate in ul/min
    
    overshoot=0 #to enable overshoot when class reached
    overshoot_time=10 # max overshoot time in seconds
    FR2_limit=12  # max flow rate for 0.7% alginate
    FR3_limit=6   # max flow rate for 2.8% alginate
    
    STD_threshold=0.4 #STD class threshold for overshoot
    
    print("now entering the loop!")
    t0=time.time() #initial time
        
    #ENTER THE GRADIENT LOOP
    
    with open('flow_rates.csv', 'w', newline='') as csv_file: 
        csv_writer = csv.writer(csv_file,dialect='excel') #delimiter=','
        l=[['Time (s)'],['Flow rate 1'],['Flow rate 2'],['Flow rate 3'],['Target vol 1'],
            ['Target vol 2'],['Target vol 3'],['dosed 1'],['dosed 2'],['dosed 3'],
            ['CNN Class'],['CNN Proba'],['Class STD']]
        l2=zip(*l) #needed to save data as columns
        csv_writer.writerows(l2) # labels per colum
    
        pump1.generate_flow(oil_FR) #generate flow for oil during whole experiment same
        
        while j!=300: # number of seconds to run screen in total    
                        
            j+=1
            
            if overshoot==0:
                
                #Acquire 50 frames
                for i in range (50):
        
                    frame0.waitFrameCapture(timeout=2147483648)   # wait for next triggered frame
                    frame0.queueFrameCapture()
                   
                    data1_np = np.ndarray(buffer=frame0.getBufferByteData(), #store image
                              dtype=np.uint8,
                              shape=(frame0.height, frame0.width))
                     
                     
                    data2_np=data1_np.astype(np.float32)
        
                    data2_np=data2_np-np.mean(data2_np) #normalization
                    data2_np=data2_np/np.std(data2_np) #normalization
                    
                    data2_np = np.reshape(data2_np, (1,120,120))
                    data2_np = np.expand_dims(data2_np, axis = 3)  
        
                    predictions = sess.run(softmax_tensor, {'tf_reshape1:0': data2_np} ) # Passes the image data (data2_np) through the graph
        
                    cnn_result_50[i] = np.argmax(predictions[0]) # Predictions contains the probability for the two classes. Max probabilty is selected as the predicted result
                    cnn_proba_50[i] = np.max(predictions[0])
                    
                    #saving image
                    imageio.imsave(os.path.join('Image Collection\Live',
                                     'foo{} '.format(50*(j-1)+i)+ str(cnn_result_50[i] )+' '+str(round(cnn_proba_50[i],2))+'.png'),data1_np.astype('uint8')) #with saving
                      
                
                t1=time.time()-t0 #get local time
                
                #print current class and probability
                print("current class = {}, proba = {}".format(np.mean(cnn_result_50), 
                      "{0:.2f}".format(np.mean(cnn_proba_50))))     
                                   
                FLOW1=pump1.get_flow_is() #get current flow rate for pump 1
                FLOW2=pump2.get_flow_is() #get current flow rate for pump 2
                FLOW3=pump3.get_flow_is() #get current flow rate for pump 3
                
                if step>0: #step is step increase/decrease in flow rate and is either a positive or negative number
                
                    
                    if FLOW2<FR2_limit: # hard limit on flow rate 
                        pump2.generate_flow(FLOW2+step) #add step to 0.7% alginate flow rate
                    
                    if FLOW3-3*step>=0: #ensure flows stays at or above 0 ul/min
                        pump3.generate_flow(FLOW3-3*step) #subtract 3 steps to 3% alginate flow rate
                
                else:
                
                    if FLOW2>0: # hard limit on flow rate 
                        pump2.generate_flow(FLOW2+step) #add step to 0.7% alginate flow rate
                   
                    if FLOW3<FR3_limit: #hard limit on flow rate
                        pump3.generate_flow(FLOW3-step) #subtract step to 3% alginate flow rate
                                
                vol1=pump1.get_target_volume()
                vol2=pump2.get_target_volume()
                vol3=pump2.get_target_volume()
                
                dos1=pump1.get_dosed_volume()
                dos2=pump2.get_dosed_volume()
                dos3=pump2.get_dosed_volume()
                
                #display newly updated flow rates
                print("time =", "{0:.2f}".format(t1), "Flow 1 =","{0:.2f}".format(FLOW1), "Flow 2 =", "{0:.2f}".format(FLOW2), "Flow 3 ="
                      "{0:.2f}".format(FLOW3))  
             	
                #record the data in a csv file
                l=[[t1],[FLOW1],[FLOW2],[FLOW3],[vol1],[vol2],[vol3],[dos1],[dos2],[dos3],[np.mean(cnn_result_50)],[np.mean(cnn_proba_50)],[np.std(cnn_result_50)]]
                l2=zip(*l) #needed to save data as columns
                csv_writer.writerows(l2) # labels per column     
       
                
                #Overshoot function
                if (target_class==0 and np.mean(cnn_result_50)<(1-target_thresh)):
                     target_class=abs(target_class-1) # reverse target class
                     overshoot=1
                     
                     
                elif (target_class==1 and np.mean(cnn_result_50)>target_thresh):
                     target_cla95ss=abs(target_class-1) # reverse target class
                     overshoot=1
                     
                          
            else:
                
                print('entering overshoot')
                
                while (overshoot!=0):
                    
                    #Acquire 50 frames
                    for i in range (50):
            
                        frame0.waitFrameCapture(timeout=2147483648)
                        frame0.queueFrameCapture()
                       
                        data1_np = np.ndarray(buffer=frame0.getBufferByteData(), #store image
                                  dtype=np.uint8,
                                  shape=(frame0.height, frame0.width))
                         
                         
                        data2_np=data1_np.astype(np.float32) # convert to float
            
                        data2_np=data2_np-np.mean(data2_np) #normalize
                        data2_np=data2_np/np.std(data2_np)
                        
                        data2_np = np.reshape(data2_np, (1,120,120))
                        data2_np = np.expand_dims(data2_np, axis = 3)  
            
                        predictions = sess.run(softmax_tensor, {'tf_reshape1:0': data2_np} ) # Passes the image data through the graph
            
                        cnn_result_50[i] = np.argmax(predictions[0]) # Predictions contains the probability for the two classes. Max probabilty is selected as the predicted result
                        cnn_proba_50[i] = np.max(predictions[0])
                    
                        imageio.imsave(os.path.join('Image Collection\Live',
                                         'foo{} '.format(50*(j-1)+i)+ str(cnn_result_50[i] )+' '+str(round(cnn_proba_50[i],2))+'.png'),data1_np.astype('uint8')) #with saving
                          
                    
                    t1=time.time()-t0 #get local time and save
                    
                    #print current class and probability
                    print("current class = {}, proba = {}".format(np.mean(cnn_result_50), 
                          "{0:.2f}".format(np.mean(cnn_proba_50))))  
                    print("current class STD =","{0:.2f}".format(np.std(cnn_result_50)))
                                       
                    FLOW1=pump1.get_flow_is() #get current flow rate for pump 1
                    FLOW2=pump2.get_flow_is() #get current flow rate for pump 2
                    FLOW3=pump3.get_flow_is() #get current flow rate for pump 3
                    
                    if step>0: 
                
                    
                        if FLOW2<FR2_limit: # hard limit on flow rate 
                            pump2.generate_flow(FLOW2+step) #update flow for 0.7% alginate
                        
                        if FLOW3-3*step>=0: #stay at 0ul/mi if already there
                            pump3.generate_flow(FLOW3-3*step) #update flow for 3% alginate 
                    
                    else:
                
                        if FLOW2>0: # hard limit on flow rate 
                            pump2.generate_flow(FLOW2+step) #update flow for 0.7% alginate
                        if FLOW3<FR3_limit: #hard limit on flow rate
                            pump3.generate_flow(FLOW3-step) #update flow for 3% alginate
                        
                    
                    vol1=pump1.get_target_volume()
                    vol2=pump2.get_target_volume()
                    vol3=pump2.get_target_volume()
                    
                    dos1=pump1.get_dosed_volume()
                    dos2=pump2.get_dosed_volume()
                    dos3=pump2.get_dosed_volume()
                    
                    print("time =", "{0:.2f}".format(t1), "Flow 1 =","{0:.2f}".format(FLOW1), "Flow 2 =", "{0:.2f}".format(FLOW2), "Flow 3 ="
                          "{0:.2f}".format(FLOW3))  
                 	      
                    l=[[t1],[FLOW1],[FLOW2],[FLOW3],[vol1],[vol2],[vol3],[dos1],[dos2],[dos3],[np.mean(cnn_result_50)],[np.mean(cnn_proba_50)],[np.std(cnn_result_50)]]
                    l2=zip(*l) #needed to save data as columns
                    csv_writer.writerows(l2) # labels per colum     
            
                    overshoot+=1
                    print('Overshoot',overshoot)
            
                    #condition for exiting overshoot:STD of CLASS is HIGH
                    if  ((np.std(cnn_result_50)>STD_threshold) or overshoot>overshoot_time ):
                       
                        overshoot=0 # reinitialize overshoot
                        step=-step  #reverse flow rate steps
                        print('Overshoot finished, new target_class',target_class)
            
            
            