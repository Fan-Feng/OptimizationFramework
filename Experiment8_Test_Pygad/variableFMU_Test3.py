import os, sys
from pyfmi import load_fmu
import numpy as np
import time

start_time=0
final_time=60*60*24*31

#load a new fmu file 
model_temp = load_fmu(fmu = "ASHRAE901_OfficeMedium_STD2016_NewYorkOccCou_HVAC_9_3_Bias_EMS_FMUVariable.fmu",log_level = 4)

## model initilization 
model_temp.setup_experiment(False, 0, 0, True, final_time)
model_temp.initialize(start_time, final_time)

opts = model_temp.simulate_options()
idf_steps_per_hour = 12
ncp = int(final_time/(3600./idf_steps_per_hour))
opts['ncp'] = ncp

tim = start_time
outputs = []
i = 0
while tim < final_time:
    model_temp.set('CONFROOM_SENSOR_BIASFMU',1)
    for j in range(ncp):
        model_temp.do_step(tim, 3600./idf_steps_per_hour, True)
        output = model_temp.get ('TRooMea')
        outputs.append(output)
        tim = tim + 3600./idf_steps_per_hour
    i += 1
model_temp.terminate()
