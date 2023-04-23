#!/usr/bin/env python
# coding: utf-8

# ## Importation of Libraries and Definition of Essential Functions

# ### Libraries 

# In[1]:


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


# ### Functions

# In[2]:


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


# **Objective:** Collects serial data from the normally-open, normally-closed, and BendLabs sensors, which collectively function as a Multi-Functional Open and Closed Sensor (MFOC-S). After reading a line from the serial output, it checks to see if the full data has been recieved at the current timestep; it will go into a while loop where it conducts the data collection if it determines that it is at the beginning of the sentence containing the serial values; otherwise, there would be a '\\n'. After entering the while loop, 

# In[3]:


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


# **Objective:** Text Processing to extract data into arrays
# 

# In[215]:


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
        radius = -(0.5*c)/math.sin(theta*(math.pi/180)) + 0.002
        strain = (0.5*thickness/radius)
    else:
      theta = 90
      radius = np.nan
      strain = 0
      
    return [x, radius, strain] 


# **Objective:** Function recieves the contact angle and position data and outputs position, radius, and stain.
# 

# In[5]:


def blss(s):
    strain = s/100 
    E = 3600000 #in Pa
    stress = strain*E
    return [strain, stress]


# **Objective:** #function takes in bendlabs percent strain and outputs strain and stress
# 

# ## Data Reading, Cleaning, and Wrangling

# ### Data Reading and Cleaning

# In[6]:


ser = serial.Serial('/dev/tty.usbmodem14401', 9600, timeout=1)
time.sleep(2)
print("connected to: " + ser.portstr)


# In[9]:


t0 = time.time()
collect_data(ser, t0)


# In[104]:


sample = []
sample_time = float(input("Enter a collection time in minutes:"))
t0 = time.time()
###################################### MAIN DATA COLLECTION WHILE LOOP #########################################################
while (((time.time() - t0)/60) < sample_time):
    sample.append(collect_data(ser, t0))
sample


# In[105]:


processed = process_data(sample, 7)
print("Contact Sensor Region-Specific Angle Data:", processed[0][0])
print("BendLabs Sensor Data:", processed[1][0])
print("Time Data:", processed[2][0])


# ### Data Wrangling

# In[12]:


# Determine Time it takes to execute the program herein
start_time = time.time()

# Principal DataFrames' Creation

    # DataFrame for Contact Sensors
ocsensors = pd.DataFrame({"Sensor Type" : [],
              "Sensor Region" : [] ,
              "Position (mm)" : [],
              "Contact Angle" : [],
              "Radius of Curvature (mm)" : [],
              "Bending Strain" : [],
              "Time (s)" : [] })

    # DataFrame for BendLabs Sensor 
blsensor = pd.DataFrame({ "In-Plane Strain" : [] ,
              "Stress (Pa)" :[] ,
              "Time (s)" :[] } )

# Example of the Live-Simulation Algorithm 
for timestep in range(len(processed[2])):
    for index, contactangle in enumerate(processed[0][timestep], start=1):   # default is zero
        
        # DataFrame for Contact Sensors
        ocsensors_temp = pd.DataFrame({"Sensor Type" : ['Normally Open Sensor' if any((True for i in processed[0][timestep] if i >= 0.0)) == True else 'Normally Closed Sensor'],
              "Sensor Region" : [index],
              "Position (mm)" : [index*15],
              "Contact Angle" : [contactangle],
              "Radius of Curvature (mm)" : [ocsensorf(index,contactangle)[1]],
              "Bending Strain" : [ocsensorf(index,contactangle)[2]],
              "Time (s)" : processed[2][timestep] })
    
        # DataFrame for BendLabs Sensor 
        blsensor_temp = pd.DataFrame({ "In-Plane Strain" : [processed[1][timestep][0]],
              "Stress (Pa)" : [processed[1][timestep][1]],
              "Time (s)" : processed[2][timestep]} )
        
        ocsensors = pd.concat([ocsensors, ocsensors_temp], ignore_index=True, sort=False)
        blsensor = pd.concat([blsensor, blsensor_temp], ignore_index=True, sort=False)

       # ocsensors = ocsensors.append(ocsensors_temp, ignore_index=True,sort=False)
       # blsensor =  blsensor.append(blsensor_temp, ignore_index=True,sort=False)
        
# Print Time it took to run program above
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %s seconds per row ---" % (((time.time() - start_time)/len(ocsensors))))


# ## MFOC-S DataFrames 

# ### Normally-Open and Normally-Closed Contact Sensors' Data

# In[13]:


ocsensors


# ### BendLabs Sensors' Data

# In[14]:


blsensor


# ## Live-Time Data Visualization (FuncAnimation)

# In[15]:


processed = process_data(sample, 7)
print("Contact Sensor Region-Specific Angle Data:", processed[0][0])
print("BendLabs Sensor Data:", processed[1][0])
print("Time Data:", processed[2][0])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'qt')

# Start Time Count 
t0 = time.time()

# This function is called periodically from FuncAnimation
def my_function(i):
    # Read and Process Data from Port
    time_live.popleft()
    average_bendstrain.popleft()
    bl_stress.popleft()

       # Average Bending Strain of MFOC-S [VARIABLE 1]
    sample1 = []
    while len(sample1) < 1:
        sample1.append(collect_data(ser, t0))
        if len(sample1) == 1:
            npsensordata = process_data(sample1,7)
            bap = [ocsensorf(index,contactangle)[2] for index, contactangle in enumerate(npsensordata[0][0], start=1)]
            average_bendstrain.append(sum(bap)/len(bap))   # AVERAGE BENDING STRAIN
    sample1 = []
    
        # TIME and STRESS [VARIABLES 1 AND 2]
    time_live.append(npsensordata[2][0][0])  # TIME
    bl_stress.append(npsensordata[1][0][1]) # STRESS from BendLabs Sensor

    # Clear Axes
    ax.clear()
    ax1.clear()
    
    # Rolling Window Size
    repeat_length = 10

   # ax.set_xlim([0,repeat_length])
    ax.set_ylim([-0.001,0.001])
    #ax1.set_xlim([0,repeat_length])
    ax1.set_ylim([-20,5])
    
    # plot stress and strain against time
    ax.plot(average_bendstrain)
    ax.scatter(len(average_bendstrain)-1, average_bendstrain[-1])
    ax.text(len(average_bendstrain)-1, average_bendstrain[-1], "{}%".format(average_bendstrain[-1]))

    ax1.plot(bl_stress)
    ax1.scatter(len(bl_stress)-1, bl_stress[-1])
    ax1.text(len(bl_stress)-1, bl_stress[-1], "{}%".format(bl_stress[-1]))



# start collections with zeros
time_live = collections.deque(np.zeros(10))
average_bendstrain = collections.deque(np.zeros(10))
bl_stress = collections.deque(np.zeros(10))

# define and adjust figure
fig = plt.figure(figsize=(12,6), facecolor='#DEDEDE')
ax = plt.subplot(121)
ax1 = plt.subplot(122)
ax.set_facecolor('#DEDEDE')
ax1.set_facecolor('#DEDEDE')

# animate
ani = FuncAnimation(fig, my_function, interval=1000, blit = False)
plt.show()


# ## Live StreamLit Dashboard 

# In[19]:


process_data(sample,7)


# In[54]:


collect_data(ser, t0)


# In[141]:


for i in range(5):
    sample = []
    sery = collect_data(ser, t0)
    sample.append(sery)
    y = process_data(sample,7)
    print(y,sery)


# In[211]:


# flights_sample1[flights_sample1.columns.difference(['month','day','day_of_week'])].
#    sort_values(by=['day_of_year']).groupby(['day_of_year']).mean().reset_index()
ocsensors1.tail(10).groupby(['Sensor Region']).mean()['Bending Strain'].sum()/7


# In[188]:


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

t0 = time.time()
while True: 
    sample = []
    while len(sample) < 1:
        sample.append(collect_data(ser, t0))
        if len(sample) == 1:
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
            print(blsensor1)
    time.sleep(0.25)


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
        sample.append(collect_data(ser, t0))
        if len(sample) == 1:
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

