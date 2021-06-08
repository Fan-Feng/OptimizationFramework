import subprocess as sp

sp.call(["energyplus", "-w","USA_CO_Golden-NREL.724666_TMY3.epw", "-r","1ZoneUncontrolled.idf"])

print("Simulation completed successfully")