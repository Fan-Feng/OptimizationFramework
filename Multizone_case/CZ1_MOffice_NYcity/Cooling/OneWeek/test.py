import pandas as pd

def read_result(filename):
  import datetime
  ## a function used to process ESO file

  output_idx = [675,676,677,678,679,1741,1747,1753,1760,1766,1999,2041] # Indices for  
  data = {'dtime':[],
          'dayType':[]}
  for id_i in output_idx:
    data[str(id_i)] = []

  with open(filename) as fp:
    while True:
      line = fp.readline()
      ## Skip the header part
      if line.startswith('End of Data Dictionary'):
        break
      else:
        continue
      
    while True:
      line = fp.readline()
      if line.startswith('End of Data'):
        break

      fields = [f.strip() for f in line.split(',')]
      id = int(fields[0])
      if id == 2: # this is the timestamp for all following outputs
        dtime = datetime.datetime(2021,int(fields[2]),int(fields[3]),int(float(fields[5]))-1,int(float(fields[6])))
        dayType = fields[-1]
        data['dtime'].append(dtime)
        data['dayType'].append(dayType)
        continue

      if id in output_idx:
        data[str(id)].append(float(fields[1]))
      else:
        # skip entries that are not output:variables
        continue
  
  data = pd.DataFrame(data)
  print(data.iloc[:2,:])
  return data

dat = read_result('MediumOff_NewYork.eso')