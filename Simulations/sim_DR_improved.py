## DISPATCHING RULE
# Parameters SETTINGS
max_gen_ene = 250
population_number_ene = 20

# Parameters SETTINGS
max_gen_mks = 250
population_number_mks = 20

import random
import numpy as np


def compute_best_indices_mks(filename):
    import pandas as pd
    from collections import Counter
    # Initialize the TP dictionary
    TP = {}
    # Initialize the list to store the number of operations for each job
    cont = []

    # Import data from Excel file
    df = pd.read_excel(filename, sheet_name='Processing times')

    # Extract the number of machines from the 'summary' sheet
    summary_df = pd.read_excel(filename, sheet_name='Summary')
    m = summary_df['Number of Machines'].values[0]
    n = summary_df['Number of Jobs'].values[0]

    # Loop through the DataFrame rows to populate the TP dictionary
    for index, row in df.iterrows():
        job = row['JOB']
        operation = row['OPERATION']
        for i in range(1, m + 1):
            # Processing times of (job, assignment, machine)
            if row[f"M{i}"] != 1000:
                TP[(job, operation, i)] = row[f"M{i}"]
        # Check if the current job is not in the list N and append the number of operations for that job
        cont.append(job)

    N = dict(Counter(cont))

    # Generation of dictionary of eligible machine for assignment ij
    M_ij = {}
    for key in TP.keys():
        job, operation, machine = key
        if (job, operation) not in M_ij:
            M_ij[(job, operation)] = []
        M_ij[(job, operation)].append(machine)

    # Initialize transportation times
    df_trt = pd.read_excel(filename, sheet_name='Transportation times')
    TRT = {}
    # Loop through the DataFrame rows and columns to populate the TRT dictionary
    for q in range(0, m + 1):
        for k in range(0, m + 1):
            TRT[(q, k)] = df_trt.iloc[q, k + 1]

    # Energy consumption from Other power
    df_en = pd.read_excel(filename, sheet_name='Other power')
    IDLE_power = []
    for m in range(1, m + 1):
        IDLE_power.append(df_en.iloc[0, m])
    AUX_power = df_en.iloc[1, 1]  # kW/h
    AGV_power = df_en.iloc[2, 1]  # kW/h

    ## Production power extraction
    df_prod = pd.read_excel(filename, sheet_name='Production power')
    PP = {}
    for index, row in df_prod.iterrows():
        job = row['JOB']
        operation = row['OPERATION']
        for i in range(1, m + 1):
            # Processing times of (job, assignment, machine)
            PP[(job, operation, i)] = row[f"M{i}"]

    # Total number of operations
    tot_number_operations = len(cont)
    # Sequence of job [1 1 1 2 2 3 3 3] the number of time of assignment
    sequence_job = cont
    import time
    ## Mixed Dispatching Rule Meta-heuristic algorithm
    scheduled_jobs_orig = [x for x in TP.keys()]

    M = list(range(1, m + 1))
    I = list(range(1, n + 1))
    O_ij = {job: list(range(1, N[job] + 1)) for job in range(1, n + 1)}
    T_ijm = TP
    total_Operation = tot_number_operations

    sorted_M_ij = {}
    for job_op, machines in M_ij.items():
        sorted_machines = sorted(machines, key=lambda machine: TP.get((job_op[0], job_op[1], machine)))
        sorted_M_ij[job_op] = sorted_machines

    # Compute the difference in processing time for sequential machines in sorted_M_ij
    diff_best = {}
    for job_op, machines in sorted_M_ij.items():
        for i, machine in enumerate(machines):
            # Calculate the difference in processing times between this machine and the best one
            current_time = TP[(job_op[0], job_op[1], machine)]
            best_time = TP[(job_op[0], job_op[1], machines[0])]
            diff_best[(job_op[0], job_op[1], machine)] = current_time - best_time

    scheduled_jobs_orig = [x for x in TP.keys()]
    total_Operation = tot_number_operations
    iteration = []
    history = []
    gen = 0
    current_min_makespan = 10000

    initial_t = time.time()

    while gen < max_gen_mks:
        print(gen)

        # initialisation available operations
        scheduled_jobs = [jobs for jobs in scheduled_jobs_orig if jobs[1] == 1]

        scheduling = []
        operation_done = {job: 0 for job in range(1, n + 1)}
        precedence_machine = {job: 0 for job in range(1, n + 1)}
        finish_time_machine = {ma: 0 for ma in range(1, m + 1)}
        finish_time_job = {j: 0 for j in range(1, n + 1)}

        # First decision, first ranking of available jobs
        rand = random.random()
        if rand < 0.8:
            scheduled_jobs = sorted(scheduled_jobs, key=lambda x: (diff_best.get(x), random.random()))
        else:
            # Shortest processing time job applied to the shortest processing time machine
            scheduled_jobs = sorted(scheduled_jobs, key=lambda x: (TP[x], random.random()))

        while len(scheduled_jobs) != 0:
            for mission in scheduled_jobs:
                if len(scheduled_jobs) != 0:
                    job = mission[0]
                    machine = mission[2]

                    # eligible_machine is the list of machines that could be assigned considering current availability
                    # eligible_job the list of jobs that could be assigned on the machine we want to assign considering current availability
                    eligible_machines = []
                    eligible_jobs = []
                    for other_missions in scheduled_jobs:
                        if other_missions[2] not in eligible_machines:
                            eligible_machines.append(other_missions[2])
                        if other_missions[2] == machine and other_missions[0] not in eligible_jobs:
                            eligible_jobs.append(other_missions[0])

                    # Adjusting the if condition to work with lists
                    if all(finish_time_machine.get(other_eligible_machine) >= finish_time_machine[machine] for
                           other_eligible_machine in eligible_machines) and (all(
                            finish_time_job[other_eligible_job] + TRT[
                                precedence_machine[other_eligible_job], machine] >=
                            finish_time_job[job] + TRT[precedence_machine[job], machine] for other_eligible_job
                            in eligible_jobs) or finish_time_job[job] + TRT[precedence_machine[job], machine] <=
                                                                             finish_time_machine[machine]):

                        operation_number = mission[1]

                        # Precedence machine of the job
                        machine_pre = precedence_machine.get(job)
                        transport = TRT[(machine_pre, machine)]
                        real_transport = TRT[(machine_pre, machine)]
                        # Compute transportation time interference
                        if machine in finish_time_machine or job in finish_time_job:
                            if machine in finish_time_machine and job in finish_time_job:
                                start_time = max(finish_time_job[job], finish_time_machine[machine])
                            else:
                                if machine in finish_time_machine:
                                    start_time = finish_time_machine[machine]
                                else:
                                    start_time = finish_time_job[job]
                        else:
                            start_time = 0
                        if job in finish_time_job and start_time - finish_time_job[job] - transport <= 0:
                            real_transport = - start_time + finish_time_job[job] + transport
                        if job in finish_time_job and start_time - finish_time_job[job] - transport > 0:
                            real_transport = 0

                        # Update all variable
                        scheduling.append([machine, operation_number, job])
                        finish_time_job[job] = start_time + real_transport + TP[(job, operation_number, machine)]
                        precedence_machine[job] = machine
                        operation_done[job] = operation_number
                        finish_time_machine[machine] = start_time + real_transport + TP[
                            (job, operation_number, machine)]

                        # Update available operations
                        if N[job] != operation_number:
                            for mach in M_ij[job, operation_number + 1]:
                                scheduled_jobs.append((job, operation_number + 1, mach))

                        for mach in M_ij[job, operation_number]:
                            scheduled_jobs.remove((job, operation_number, mach))

                        # Next Dispatching Rule Choice
                        rand = random.random()
                        if rand < len(
                                scheduling) / tot_number_operations:  # % NOR probability increases during the scheduling assignment
                            # Sort the list based on the number of operations in descending order
                            scheduled_jobs = sorted(scheduled_jobs, key=lambda x: (
                            N[x[0]] - operation_done[x[0]], -diff_best.get(x), random.random()),
                                                    reverse=True)
                        elif rand < len(scheduling) / tot_number_operations + 0.1 * (
                                1 - len(scheduling) / tot_number_operations):
                            # Sort the list based on SPT
                            scheduled_jobs = sorted(scheduled_jobs, key=lambda x: (TP[x], random.random()))
                        else:
                            # (J,O,M) ranked based on difference between PT on that machine and best PT for that operation
                            scheduled_jobs = sorted(scheduled_jobs, key=lambda x: (diff_best.get(x), random.random()))

                        break

        total_times = {machine: time + TRT[(machine, 0)] for machine, time in finish_time_machine.items()}
        total_makespan = max(total_times.values())
        if total_makespan < current_min_makespan:
            current_min_makespan = total_makespan
        history.append([scheduling, total_makespan])
        iteration.append([time.time() - initial_t, current_min_makespan])
        gen += 1

    # encoding best scheduling
    os_job = []
    dr_mix_population = []
    job_operation_to_machine = []
    history.sort(key=lambda x: x[1])
    for i in range(max_gen_mks):
        os_job.append([])
        for job in history[i][0]:
            os_job[i].append(job[2])
        job_operation_to_machine.append({(entry[2], entry[1]): entry[0] for entry in history[i][0]})

    for i in range(max_gen_mks):
        os_machine = []
        for job in range(1, n + 1):
            for operation in range(1, N[job] + 1):
                os_machine.append(job_operation_to_machine[i].get((job, operation)))

        dr_mix_population.append(os_job[i] + os_machine)

    Population_mks = np.array(dr_mix_population)

    final_t = time.time()
    print("Total time", final_t - initial_t)

    return Population_mks, population_number_mks, iteration, diff_best


def compute_best_indices_ene(filename):

    import pandas as pd
    from collections import Counter
    # Initialize the TP dictionary
    TP = {}
    # Initialize the list to store the number of operations for each job
    cont = []

    # Import data from Excel file
    df = pd.read_excel(filename, sheet_name='Processing times')

    # Extract the number of machines from the 'summary' sheet
    summary_df = pd.read_excel(filename, sheet_name='Summary')
    m = summary_df['Number of Machines'].values[0]
    n = summary_df['Number of Jobs'].values[0]

    # Loop through the DataFrame rows to populate the TP dictionary
    for index, row in df.iterrows():
        job = row['JOB']
        operation = row['OPERATION']
        for i in range(1, m + 1):
            # Processing times of (job, assignment, machine)
            if row[f"M{i}"] != 1000:
                TP[(job, operation, i)] = row[f"M{i}"]
        # Check if the current job is not in the list N and append the number of operations for that job
        cont.append(job)

    N = dict(Counter(cont))

    # Generation of dictionary of eligible machine for assignment ij
    M_ij = {}
    for key in TP.keys():
        job, operation, machine = key
        if (job, operation) not in M_ij:
            M_ij[(job, operation)] = []
        M_ij[(job, operation)].append(machine)

    # Initialize transportation times
    df_trt = pd.read_excel(filename, sheet_name='Transportation times')
    TRT = {}
    # Loop through the DataFrame rows and columns to populate the TRT dictionary
    for q in range(0, m + 1):
        for k in range(0, m + 1):
            TRT[(q, k)] = df_trt.iloc[q, k + 1]

    # Energy consumption from Other power
    df_en = pd.read_excel(filename, sheet_name='Other power')
    IDLE_power = []
    for m in range(1, m + 1):
        IDLE_power.append(df_en.iloc[0, m])
    AUX_power = df_en.iloc[1, 1]  # kW/h
    AGV_power = df_en.iloc[2, 1]  # kW/h

    ## Production power extraction
    df_prod = pd.read_excel(filename, sheet_name='Production power')
    PP = {}
    for index, row in df_prod.iterrows():
        job = row['JOB']
        operation = row['OPERATION']
        for i in range(1, m + 1):
            # Processing times of (job, assignment, machine)
            PP[(job, operation, i)] = row[f"M{i}"]

    # Total number of operations
    tot_number_operations = len(cont)
    # Sequence of job [1 1 1 2 2 3 3 3] the number of time of assignment
    sequence_job = cont
    import time

    scheduled_jobs_orig = [x for x in TP.keys()]
    M = list(range(1, m + 1))
    I = list(range(1, n + 1))
    O_ij = {job: list(range(1, N[job] + 1)) for job in range(1, n + 1)}
    T_ijm = TP

    total_Operation = tot_number_operations

    initial_time = time.time()
    # energy consumed by a set of (job,assignment,machine) is given as PROD. TIME * PROD.POWER + PAUX * PROC. TIME / N_MACHINES
    energy = {}
    for element in scheduled_jobs_orig:
        energy[element] = round(TP[element] / 60 * PP[element], 2)

    sorted_M_ij = {}
    for job_op, machines in M_ij.items():
        sorted_machines = sorted(machines, key=lambda machine: energy.get((job_op[0], job_op[1], machine)))
        sorted_M_ij[job_op] = sorted_machines

    # Compute the difference in processing time for sequential machines in sorted_M_ij
    diff_dict = {}
    for job_op, machines in sorted_M_ij.items():
        for i, machine in enumerate(machines):
            # Calculate the difference in processing times between this machine and the previous one
            current_energy = energy[(job_op[0], job_op[1], machine)]
            best_energy = energy[(job_op[0], job_op[1], machines[0])]
            diff_dict[(job_op[0], job_op[1], machine)] = round(current_energy - best_energy, 1)
    ## Mixed Dispatching Rule Meta-heuristic algorithm

    history = []
    gen = 0
    iteration_energy = []
    while gen < max_gen_ene:
        print(gen)

        # initialisation available operations
        scheduled_jobs = [jobs for jobs in scheduled_jobs_orig if jobs[1] == 1]

        scheduling = []
        operation_done = {job: 0 for job in range(1, n + 1)}
        precedence_machine = {job: 0 for job in range(1, n + 1)}
        finish_time_machine = {machine: 0 for machine in range(1, m + 1)}
        finish_time_job = {j: 0 for j in range(1, n + 1)}

        # First decision, first ranking of available jobs
        rand = random.random()
        if rand < 0.8:
            scheduled_jobs = sorted(scheduled_jobs, key=lambda x: (diff_dict.get(x), random.random()))
        else:
            # Shortest processing time job applied to the shortest processing time machine
            scheduled_jobs = sorted(scheduled_jobs, key=lambda x: (TP[x], diff_dict.get(x), random.random()))

        while len(scheduled_jobs) != 0:
            for mission in scheduled_jobs:
                if len(scheduled_jobs) != 0:
                    job = mission[0]
                    machine = mission[2]

                    # eligible_machine is the list of machines that could be assigned considering current availability
                    # eligible_job the list of jobs that could be assigned on the machine we want to assign considering current availability
                    eligible_machines = []
                    eligible_jobs = []
                    for other_missions in scheduled_jobs:
                        if other_missions[2] not in eligible_machines:
                            eligible_machines.append(other_missions[2])
                        if other_missions[2] == machine and other_missions[0] not in eligible_jobs:
                            eligible_jobs.append(other_missions[0])

                    # Adjusting the if condition to work with lists
                    if all(finish_time_machine.get(other_eligible_machine) >= finish_time_machine[machine] for
                           other_eligible_machine in eligible_machines) and (all(
                        finish_time_job[other_eligible_job] + TRT[precedence_machine[other_eligible_job], machine] >=
                        finish_time_job[job] + TRT[precedence_machine[job], machine] for other_eligible_job
                        in eligible_jobs) or finish_time_job[job] + TRT[precedence_machine[job], machine] <=
                                                                             finish_time_machine[machine]):

                        operation_number = mission[1]

                        # Precedence machine of the job
                        machine_pre = precedence_machine[job]
                        transport = TRT[(machine_pre, machine)]
                        real_transport = TRT[(machine_pre, machine)]

                        if machine in finish_time_machine or job in finish_time_job:
                            if machine in finish_time_machine and job in finish_time_job:
                                start_time = max(finish_time_job[job], finish_time_machine[machine])
                            else:
                                if machine in finish_time_machine:
                                    start_time = finish_time_machine[machine]
                                else:
                                    start_time = finish_time_job[job]
                        else:
                            start_time = 0
                        if job in finish_time_job and start_time - finish_time_job[job] - transport <= 0:
                            real_transport = - start_time + finish_time_job[job] + transport
                        if job in finish_time_job and start_time - finish_time_job[job] - transport > 0:
                            real_transport = 0

                        # update all variables
                        scheduling.append([machine, operation_number, job])
                        finish_time_job[job] = start_time + real_transport + TP[(job, operation_number, machine)]
                        precedence_machine[job] = machine
                        operation_done[job] = operation_number
                        finish_time_machine[machine] = start_time + real_transport + TP[
                            (job, operation_number, machine)]

                        # update available operations
                        if N[job] != operation_number:
                            for mach in M_ij[job, operation_number + 1]:
                                scheduled_jobs.append((job, operation_number + 1, mach))

                        for mach in M_ij[job, operation_number]:
                            scheduled_jobs.remove((job, operation_number, mach))

                        # Next Dispatching Rule Choice
                        rand = random.random()
                        if rand < len(
                                scheduling) / tot_number_operations:  # % NOR probability increases during the scheduling assignment
                            # Sort the list based on the number of operations in descending order
                            scheduled_jobs = sorted(scheduled_jobs, key=lambda x: (
                                N[x[0]] - operation_done[x[0]], -diff_dict.get(x), random.random()),
                                                    reverse=True)
                        elif rand < len(scheduling) / tot_number_operations + 0.1 * (
                                1 - len(scheduling) / tot_number_operations):
                            # Sort the list based on SPT
                            scheduled_jobs = sorted(scheduled_jobs, key=lambda x: (TP[x], diff_dict.get(x), random.random()))
                        else:
                            # (J,O,M) ranked based on difference between PT on that machine and best PT for that operation
                            scheduled_jobs = sorted(scheduled_jobs, key=lambda x: (diff_dict.get(x), random.random()))

                        break

        total_times = {machine: time + TRT[(machine, 0)] for machine, time in finish_time_machine.items()}
        total_makespan = max(total_times.values())
        history.append([scheduling, total_makespan])
        current_time = time.time()
        iteration_energy.append([current_time - initial_time, 0])
        gen += 1

    # codifica best scheduling in codice genetico
    os_job = []
    dr_mix_population = []
    job_operation_to_machine = []

    for i in range(len(history)):
        os_job.append([])
        for job in history[i][0]:
            os_job[i].append(job[2])
        job_operation_to_machine.append({(entry[2], entry[1]): entry[0] for entry in history[i][0]})

    for i in range(len(history)):
        os_machine = []
        for job in range(1, n + 1):
            for operation in range(1, N[job] + 1):
                os_machine.append(job_operation_to_machine[i].get((job, operation)))
        dr_mix_population.append(os_job[i] + os_machine)

    Population_ene = np.array(dr_mix_population)

    return Population_ene, population_number_ene, iteration_energy, diff_dict, energy