
import os, sys
from pyfmi import load_fmu
import numpy as np
import matplotlib.pyplot as plt


#load a new fmu file 
model = load_fmu(fmu = "ASHRAE901_OfficeMedium_STD2016_NewYorkOccCou_HVAC_9_3_Bias_EMS_FMUVariable.fmu",log_level = 4)
model.set('CONFROOM_SENSOR_BIASFMU', 1)

# get options object
opts = model.simulate_options()

# set number of communication points dependent on final_time and .idf steps per hour
final_time = 60*60*24*365. 
idf_steps_per_hour = 12 # 5 min timestep
ncp = int(final_time/(3600./idf_steps_per_hour))
opts['ncp'] = ncp

res =  model.simulate(start_time=0., final_time=final_time, options=opts)

fig, ax1 = plt.subplots()
ax1.plot(res['time'], res['TRooMea'], 'b.')
#ax1.plot(res['time'], res['EGasVAVBot'], 'b-')
#ax1.plot(res['time'], res['EGasVAVMid'], 'b--')
#ax1.plot(res['time'], res['EGasVAVTop'], 'b-.')
#ax1.plot(res['time'], res['PEleBui'], 'b.')
model.terminate()
 