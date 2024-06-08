## import dataset from excel filename
'''
Parameters initialization from Excel filename in Set_parameter

m = number of machines
n = number of jobs
tot_number_operations = total number of all operations
sequence_job = [1 1 1 2 2 3 3 3] ordered sequence of operations job
N dictionary, Number of operations (job) = Ni

M_ij dictionary, Eligible machine (job,assignment) = machine k,q..
TP dictionary, Processing times (job,assignment,machine) = tpijk
TRT matrix, Transportation times between machines, (index 0 is the Load/Unload area)

PP dictionary, Production power (job,assignment,machine) = ppijk
IDLE_power list, Idle power of machine k
AGV_power
AUX_power

'''

from Set_parameters import filename
import pandas as pd
from collections import Counter
# Initialize the TP dictionary
TP = {}
# Initialize the list to store the number of operations for each job
cont = []

# Import data from Excel file
df = pd.read_excel(filename,sheet_name='Processing times')

# Extract the number of machines from the 'summary' sheet
summary_df = pd.read_excel(filename, sheet_name='Summary')
m = summary_df['Number of Machines'].values[0]
n = summary_df['Number of Jobs'].values[0]

# Loop through the DataFrame rows to populate the TP dictionary
for index, row in df.iterrows():
    job = row['JOB']
    operation = row['OPERATION']
    for i in range(1, m+1):
        # Processing times of (job, assignment, machine)
        if row[f"M{i}"] != 1000:
            TP[(job, operation, i)] = row[f"M{i}"]
    # Check if the current job is not in the list N and append the number of operations for that job
    cont.append(job)

N = dict(Counter(cont))

# Generation of dictionary of eligible machine for assignment ij
M_ij={}
for key in TP.keys():
    job, operation, machine = key
    if (job, operation) not in M_ij:
        M_ij[(job, operation)] = []
    M_ij[(job, operation)].append(machine)


# Initialize transportation times
df_trt = pd.read_excel(filename,sheet_name='Transportation times')
TRT = {}
# Loop through the DataFrame rows and columns to populate the TRT dictionary
for q in range(0, m+1):
    for k in range(0, m+1):
        TRT[(q, k)] = df_trt.iloc[q , k +1]


#Energy consumption from Other power
df_en = pd.read_excel(filename,sheet_name='Other power')
IDLE_power = []
for m in range(1,m+1):
        IDLE_power.append(df_en.iloc[0,m])
AUX_power = df_en.iloc[1,1] #kW/h
AGV_power = df_en.iloc[2,1] #kW/h


## Production power extraction
df_prod = pd.read_excel(filename,sheet_name='Production power')
PP= {}
for index, row in df_prod.iterrows():
    job = row['JOB']
    operation = row['OPERATION']
    for i in range(1, m+1):
        # Processing times of (job, assignment, machine)
        PP[(job, operation, i)] = row[f"M{i}"]

# Total number of operations
tot_number_operations=len(cont)
# Sequence of job [1 1 1 2 2 3 3 3] the number of time of assignment
sequence_job = cont