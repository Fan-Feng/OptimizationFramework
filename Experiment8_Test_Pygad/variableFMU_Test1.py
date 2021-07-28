
import os, sys
from pyfmi import load_fmu
import numpy as np
import matplotlib.pyplot as plt


#load a new fmu file 
model = load_fmu(fmu = "_fmu_export_variable.fmu",log_level = 4)

## model initilization 
final_time = 60*60*72. # 72 hour simulation
model.setup_experiment(False, 0, 0, True, final_time)
model.initialize(0, final_time)

model.set('yShadeFMU', 1)

# get options object
opts = model.simulate_options()
# set number of communication points dependent on final_time and .idf steps per hour
idf_steps_per_hour = 6
ncp = int(final_time/(3600./idf_steps_per_hour))
opts['ncp'] = ncp


res =  model.simulate(start_time=0., final_time=final_time, options=opts)

print('complete')
fig, ax1 = plt.subplots()
ax1.plot(res['time'], res['TRoo'], 'b-')

model.terminate()
 