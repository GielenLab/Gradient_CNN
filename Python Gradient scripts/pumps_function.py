# -*- coding: utf-8 -*-

@author: Fabrice Gielen

import time
import ctypes
import sys
import csv

from qmixsdk import qmixbus
from qmixsdk import qmixpump

sys.path.append(r"D:\QmixSDK\lib\python") #path for Qmix SDK python libraries

# Qmix device configuration path
config_name2=r"C:\Users\Public\Documents\QmixElements\Projects\default_project\Configurations\3syringes"


def initialize_pump(config_name):
       
    print("Enabling pump drive...")
     
    pump = qmixpump.Pump()
    pumpcount = pump.get_no_of_pumps() #get number of pumps connected
    

    for i in range (pumpcount):
                
        if i==1:
                pump1 = qmixpump.Pump()
                pump1.lookup_by_device_index(i)
                print("Name of pump ", i, " is ", pump1.get_device_name())
                if pump1.is_in_fault_state():
                    pump1.clear_fault()
        elif i==2:
                pump2 = qmixpump.Pump()
                pump2.lookup_by_device_index(i)
                print("Name of pump ", i, " is ", pump2.get_device_name())
                if pump2.is_in_fault_state():
                    pump2.clear_fault()
        elif i==3:
                pump3 = qmixpump.Pump()
                pump3.lookup_by_device_index(i)
                print("Name of pump ", i, " is ", pump3.get_device_name())
                if pump3.is_in_fault_state():
                    pump3.clear_fault()
        else:
                pump4 = qmixpump.Pump()
                pump4.lookup_by_device_index(i)
                print("Name of pump ", i, " is ", pump4.get_device_name())
                if pump4.is_in_fault_state():
                    pump4.clear_fault()
    #pump.lookup_by_device_index(1)
    
   
    pump1.enable(True)
    pump2.enable(True)
    pump3.enable(True)
    pump4.enable(True)
   
    return [pump1,pump2,pump3,pump4]



def syringe_define(pump1,pump2,pump3,pump4,dia1,dia2,dia3,dia4):
        
    print("Setting syringe diameters...")

    for i in range(3):
        if i==1:
            inner_diameter_set = dia1
            piston_stroke_set = 50
            pump1.set_syringe_param(dia1, piston_stroke_set)
            
            #syringe = pump2.get_syringe_param()
        elif i==2:
            inner_diameter_set = dia2
            piston_stroke_set = 50
            pump2.set_syringe_param(dia2, piston_stroke_set)
        elif i==3:
            inner_diameter_set = dia3
            piston_stroke_set = 50
            pump3.set_syringe_param(dia3, piston_stroke_set)
        else:
            inner_diameter_set = dia4
            piston_stroke_set = 50
            pump3.set_syringe_param(dia3, piston_stroke_set)
            
    
    print("Setting flows to ul/min...")
    pump1.set_volume_unit(qmixpump.UnitPrefix.micro, qmixpump.VolumeUnit.litres)
    pump1.set_flow_unit(qmixpump.UnitPrefix.micro, qmixpump.VolumeUnit.litres, 
                qmixpump.TimeUnit.per_minute)
    pump2.set_volume_unit(qmixpump.UnitPrefix.micro, qmixpump.VolumeUnit.litres)
    pump2.set_flow_unit(qmixpump.UnitPrefix.micro, qmixpump.VolumeUnit.litres, 
                qmixpump.TimeUnit.per_minute)
    pump3.set_volume_unit(qmixpump.UnitPrefix.micro, qmixpump.VolumeUnit.litres)
    pump3.set_flow_unit(qmixpump.UnitPrefix.micro, qmixpump.VolumeUnit.litres, 
                qmixpump.TimeUnit.per_minute)
    pump4.set_volume_unit(qmixpump.UnitPrefix.micro, qmixpump.VolumeUnit.litres)
    pump4.set_flow_unit(qmixpump.UnitPrefix.micro, qmixpump.VolumeUnit.litres, 
                qmixpump.TimeUnit.per_minute)
    
    




