#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Principal Methods and Functions 
import numpy as np # np mean, np random 
import pandas as pd # read csv, df manipulation
import time # to simulate a real time data, time loop 
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import serial
import time
import sys
import math
import csv

# Live Streamlit Dashboard
import plotly.express as px # interactive charts 
import streamlit as st # web development

from matplotlib.animation import FuncAnimation
from matplotlib import figure

# System Parameters
delay = 0.05 #200 Hz

# For Storing Data AND Creating and Animating Graphs
import collections
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt, mpld3
from matplotlib.animation import FuncAnimation
from matplotlib import figure
from IPython.display import display, clear_output

# For Creating Python Dashboard
import operator as op
import requests
import param
import panel as pn
pn.extension('tabulator')
import hvplot.pandas
import hvplot.streamz
import holoviews as hv
from holoviews.element.tiles import EsriImagery
from holoviews.selection import link_selections
from datashader.utils import lnglat_to_meters
from streamz.dataframe import PeriodicDataFrame
from ipywidgets import interact

# Other
import datetime as dt
import serial
import time
import sys
import math
import csv
import psutil


# In[ ]:


def collect_data(ser, t0):
    time.sleep(delay)                   # delay of 1ms
    val = ser.readline()                # read complete line from serial output
    while not '\\n'in str(val):         # check if full data is received. 
        # This loop is entered only if serial read value doesn't contain \n
        # which indicates end of a sentence. 
        # str(val) - val is byte where string operation to check `\\n` 
        # can't be performed
        time.sleep(delay)                # delay of 1ms 
        temp = ser.readline()           # check for serial output.
        if not not temp.decode():       # if temp is not empty.
            val = (val.decode()+temp.decode()).encode()
            # requrired to decode, sum, then encode because
            # long values might require multiple passes
    val = val.decode()                  # decoding from bytes
    val = val.strip()                   # stripping leading and trailing spaces.
    return [val, round(time.time() - t0, 3)]


# In[ ]:


def process_data(data, num_regions):
    
    np_angles = np.zeros([1,num_regions])
    np_bendlabs = np.zeros([1, 2])
    np_time = np.zeros([1, 1])
    
    for i in range(len(data)):
        try:
            angles = sample[i][0].split('|')[0].strip('()').split(',')
            angles.pop()
            for j in range(len(angles)): angles[j] = float(angles[j])

            bendlabs = sample[i][0].split("|")[1].strip("()").split("  ")
            for j in range(len(bendlabs)): bendlabs[j] = [float(bendlabs[j].split(",")[0]), float(bendlabs[j].split(",")[1])]

            bendlabs = bendlabs[0]
            np_angles = np.row_stack((np_angles, np.array(angles)))
            np_bendlabs = np.row_stack((np_bendlabs, np.array(bendlabs)))
            np_time = np.row_stack((np_time, np.array(data[i][1])))
        except:
            print("Failed on: ", i, ": ", data[i])
            pass
        
    np_angles = np.delete(np_angles, 0, 0)
    np_bendlabs = np.delete(np_bendlabs, 0, 0)
    np_time = np.delete(np_time, 0, 0)
    processed_data = [np_angles, np_bendlabs, np_time]         
    return processed_data
                                       
    # except:
    #     print("Failed on ", i, data[i])


# In[ ]:


def ocsensorf(x, a):
    thickness = 0.002
    length = 7.5
    suppangle = 180 - a
    #print(suppangle)
    smalla = 90 - a/2
    #print(smalla)
    c = math.sqrt(2*length**2 - 2*length*length*math.cos(a*(math.pi/180)))
    #print(c)
    if a > 0:
      theta = (180-a)/2
      radius = (0.5*c)/math.sin(theta*(math.pi/180)) + 0.002
      strain = (0.5*thickness/radius)
    if a < 0:
        theta = (180-abs(a))/2
        radius = -(0.5*c)/math.sin(theta*(math.pi/180)) - 0.002
        strain = (0.5*thickness/-radius)
    else:
      theta = 90
      radius = np.nan
      strain = 0
      
    return [x, radius, strain] 


# In[ ]:


def blss(s):
    strain = s/100 
    E = 3600000 #in Pa
    stress = strain*E
    return [strain, stress]


# In[ ]:


ser = serial.Serial('/dev/tty.usbmodem14401', 9600, timeout=1)
time.sleep(2)
print("connected to: " + ser.portstr)


# In[ ]:


import streamlit as st # web development
import numpy as np # np mean, np random 
import pandas as pd # read csv, df manipulation
import time # to simulate a real time data, time loop 
import plotly.express as px # interactive charts 


# Principal DataFrames' Creation

     # DataFrame for Contact Sensors
ocsensors1 = pd.DataFrame({"Sensor Type" : [],
              "Sensor Region" : [] ,
              "Position (mm)" : [],
              "Contact Angle" : [],
              "Radius of Curvature (mm)" : [],
              "Bending Strain" : [],
              "Time (s)" : [] })

    # DataFrame for BendLabs Sensor 
blsensor1 = pd.DataFrame({ "In-Plane Strain" : [] ,
              "Stress (Pa)" :[] ,
              "Time (s)" :[] } )

# StreamLit Dashboard Initilization
st.set_page_config(
    page_title = 'Real-Time Data Visualization of MFOCC Sensor',
    page_icon = '🛰',
    layout = 'wide'
)

# Dashboard title
st.title("Real-Time Data Visualization of MFOC Sensor")

# Top-level filters 
sensor_region_filter = st.selectbox("Select Sensor Region", pd.unique(ocsensors1['Sensor Region']))

# creating a single-element container.
placeholder = st.empty()

# dataframe filter 
ocsensors = ocsensors1[ocsensors1['Sensor Region']==sensor_region_filter]

# near real-time / live feed simulation 
t0 = time.time()
while True: 
    sample = []
    while len(sample) < 1:
        sery = collect_data(ser, t0)
        sample.append(sery)
        sample.append(collect_data(ser, t0))
        npsensordata = process_data(sample,7)
        for index, contactangle in enumerate(npsensordata[0][0], start=1):

                            # DataFrame for Contact Sensors 
                        ocsensors_temp1 = pd.DataFrame({"Sensor Type" : ['Normally Open Sensor' if any((True for i in npsensordata[0][0] if i >= 0.0)) == True else 'Normally Closed Sensor'],
                                      "Sensor Region" : [index],
                                      "Position (mm)" : [index*15],
                                      "Contact Angle" : [contactangle],
                                      "Radius of Curvature (mm)" : [ocsensorf(index,contactangle)[1]],
                                      "Bending Strain" : [ocsensorf(index,contactangle)[2]],
                                      "Time (s)" : npsensordata[2][0] })

                            # DataFrame for BendLabs Sensor 
                        blsensor_temp1 = pd.DataFrame({ "In-Plane Strain" : [npsensordata[1][0][0]],
                                      "Stress (Pa)" : [npsensordata[1][0][1]],
                                      "Time (s)" : npsensordata[2][0]} )

                        ocsensors1 = pd.concat([ocsensors1, ocsensors_temp1], ignore_index=True, sort=False)
                        blsensor1 = pd.concat([blsensor1, blsensor_temp1], ignore_index=True, sort=False)
    # creating KPIs 
    avg_bendingstrain = ocsensors1.tail(10).groupby(['Sensor Region']).mean()['Bending Strain'].sum()/7

    avg_planestrain = blsensor1.tail(10)['In-Plane Strain'].mean()
    
    avg_stress = blsensor1.tail(10)['Stress (Pa)'].mean()
    
    with placeholder.container():
        # create three columns
        kpi1, kpi2, kpi3 = st.columns(3)

        # fill in those three columns with respective metrics or KPIs 
        kpi1.metric(label="Average Bending Strain", value=avg_bendingstrain, delta= avg_bendingstrain - 0.05*avg_bendingstrain)
        kpi2.metric(label="Average In-Plane Strain", value= avg_planestrain, delta= -0.05*avg_planestrain + avg_planestrain)
        kpi3.metric(label="Average Stress ", value= avg_stress, delta= -0.05*avg_stress + avg_stress )

        # create two columns for charts 

        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("### First Chart")
            fig = px.line(ocsensors1, x = 'Time (s)', y = 'Bending Strain')
            st.write(fig)
        with fig_col2:
            st.markdown("### Second Chart")
            fig2 = px.histogram(data_frame = ocsensors1, x = 'Bending Strain')
            st.write(fig2)
            
        st.markdown("### Detailed Data View")
        st.dataframe(ocsensors1)
        
    time.sleep(2.00)

