## Mixed Dispatching Rule Meta-heuristic algorithm

# Parameters SETTINGS
max_gen_ene = 500
population_number_ene = 20

import copy
import random
import numpy as np
import time
from Data_reader import TP,TRT,m,n,N,tot_number_operations,M_ij,PP,AGV_power,AUX_power,IDLE_power


initial = time.time()
scheduled_jobs_orig = [x for x in TP.keys()]
M = list(range(1, m + 1))
I = list(range(1, n + 1))
O_ij = {job: list(range(1, N[job]+1)) for job in range(1, n + 1)}
T_ijm = TP

total_Operation = tot_number_operations
initial_time = time.time()
# energy consumed by a set of (job,assignment,machine) is given as PROD. TIME * PROD.POWER + PAUX * PROC. TIME / N_MACHINES
energy = {}

for element in scheduled_jobs_orig:
    print(element)
    energy[element] = round(TP[element]/60 * PP[element],2)

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

history = []
gen = 0
iteration_energy = []
while gen<max_gen_ene:
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

    while len(scheduled_jobs)!=0:
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
                    in eligible_jobs) or finish_time_job[job] + TRT[precedence_machine[job], machine] <= finish_time_machine[machine]):

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
                    finish_time_machine[machine] = start_time + real_transport + TP[(job, operation_number, machine)]

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
                        scheduled_jobs = sorted(scheduled_jobs, key=lambda x: (TP[x], random.random()))
                    else:
                        # (J,O,M) ranked based on difference between PT on that machine and best PT for that operation
                        scheduled_jobs = sorted(scheduled_jobs, key=lambda x: (diff_dict.get(x), random.random()))

                    break

    total_times = {machine: time + TRT[(machine,0)] for machine, time in finish_time_machine.items()}
    total_makespan = max(total_times.values())
    history.append([scheduling, total_makespan])
    current_time = time.time()
    iteration_energy.append([current_time - initial_time, 0])
    gen += 1


#codifica best scheduling in codice genetico
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
    for job in range(1,n+1):
        for operation in range(1,N[job]+1):
            os_machine.append(job_operation_to_machine[i].get((job,operation)))
    dr_mix_population.append(os_job[i] + os_machine)

Pop_DR = []
for i in range(len(dr_mix_population)):
    Pop_DR.append(dr_mix_population[i])
Population_ene = np.array(Pop_DR)


'''
def Decode_T(Pop_matrix):  # Do not run separately
    T_list = []
    for a in range(len(Pop_matrix)):  # For each chromosome
        T = []
        for b in range(total_Operation):
            m = Pop_matrix[:][a][total_Operation:total_Operation * 2][b]  # Machine for the current assignment
            i_j = list(M_ij.keys())[b][0]  # Get the job number for this assignment
            j_i = list(M_ij.keys())[b][1]  # Get the assignment number for this assignment
            T_total = T_ijm[i_j, j_i, m]
            T.append(T_total)
        T_list.append(T)
    T_matrix = np.array(T_list)
    return T_matrix

def Decode_OS(Pop_matrix):  # Decoding
    # The sum of the number of operations of eligiblemachine jobs before the current job
    T_matrix = Decode_T(Pop_matrix)
    O_num_list = []
    O_num = 0
    for i in I:
        O_num_list.append(O_num)
        O_num += len(O_ij[i])

    # Get the corresponding job-assignment group based on the assignment code
    O_M_T_total = []
    for a in range(len(Pop_matrix)):  # For each chromosome
        O_M_T = {}
        for b in range(total_Operation):
            O_i = Pop_matrix[:][a][0:total_Operation][b]  # OS part of each chromosome
            O_j = list(Pop_matrix[:][a][0:b + 1]).count(O_i)  # The number of times the current sequence number appears, i.e., the assignment number
            T_matrix_column = O_num_list[O_i - 1] + O_j - 1  # Column number of the current assignment arranged in positive order
            O_M = Pop_matrix[:][a][total_Operation:total_Operation * 2][T_matrix_column]  # Machine selected for the current assignment
            T_matrix_recent = T_matrix[a, T_matrix_column]  # Time required for the current assignment
            O_M_T[O_i, O_j, O_M] = T_matrix_recent  # Operations sorted by OS code and corresponding equipment fixture
        O_M_T_total.append(O_M_T)
    return O_M_T_total
def crowding_distance_sort(last_front):
    num_individuals = len(last_front)
    last_front_distance = [0.0] * num_individuals  # Initialize ordered distances

    for obj_index in range(2):
        # Get indices of individuals sorted by current objective
        sorted_indices = sorted(range(num_individuals), key=lambda i: last_front[i][obj_index])

        # Assign large distances to boundary individuals and all individuals with same value
        last_front_distance[sorted_indices[0]] += 1000
        last_front_distance[sorted_indices[-1]] += 1000

        # Calculate the range of the current objective for normalisation
        obj_min = last_front[sorted_indices[0]][obj_index]
        obj_max = last_front[sorted_indices[-1]][obj_index]
        obj_range = obj_max - obj_min if obj_max - obj_min > 0 else 1  # Avoid division by zero

        # Calculate distances for intermediate individuals
        for i in range(1, num_individuals - 1):
            distance = last_front[sorted_indices[i + 1]][obj_index] - last_front[sorted_indices[i - 1]][obj_index]
            # normalized distance assigned to the correct individual
            last_front_distance[sorted_indices[i]] += distance / obj_range
    return last_front_distance
def check_dominance(solution1, solution2):
    """
    - bool: True if solution1 dominates solution2, False otherwise.
    """
    dominates = all(s1 <= s2 for s1, s2 in zip(solution1, solution2)) and any(
        s1 < s2 for s1, s2 in zip(solution1, solution2))
    return dominates

def fast_non_dominated_sort(combined_results):
    rank_fronts = []
    # number of individuals which dominates the key
    domination_counter = {}
    # set of individuals which the key dominates
    dominated_solutions = {i: set() for i in range(len(combined_results))}
    Q = set()
    for index1, individual1 in enumerate(combined_results):
        domination_counter[index1] = 0
        for index2, individual2 in enumerate(combined_results):
            if index2 != index1:
                if check_dominance(individual1, individual2):
                    dominated_solutions[index1].add(index2)
                elif check_dominance(individual2, individual1):
                    domination_counter[index1] += 1

        if domination_counter[index1] == 0:
            Q.add(index1)
    rank_fronts.append(Q)
    i = 1
    while rank_fronts[i - 1] != set():
        Q = set()
        for index1 in rank_fronts[i - 1]:
            for index2 in dominated_solutions[index1]:
                domination_counter[index2] -= 1
                if domination_counter[index2] == 0:
                    Q.add(index2)
        i += 1
        rank_fronts.append(Q)

    return rank_fronts


def fitness(time1, energy1):
    combined_results = []
    for i in range(len(time1)):
        combined_results.append([time1[i], energy1[i]])
    fronts = fast_non_dominated_sort(combined_results)

    selected_individuals = []
    current_front = 0

    Pareto_ind = len(list(fronts[0]))

    while len(selected_individuals) < population_number_ene:

        if len(fronts[current_front]) + len(selected_individuals) <= population_number_ene:
            selected_individuals.extend(list(fronts[current_front]))
            current_front += 1
        else:
            # Sort the last Pareto front based on crowding distance
            keys_last_front = list(fronts[current_front])

            # Use keys_last_front to filter the combined_results list
            last_front_individuals = [combined_results[key] for key in keys_last_front]
            distances = crowding_distance_sort(last_front_individuals)
            sorted_last_front = [ind for _, ind in
                                 sorted(zip(distances, keys_last_front), key=lambda x: x[0], reverse=True)]

            selected_individuals.extend(
                sorted_last_front[:(population_number_ene - len(selected_individuals))])
            # rank individuals on the front based on crowding distance and peak the best ones

    return selected_individuals, Pareto_ind

def Operation_insert(key, value):
    M_arranged = {a: [] for a in M}
    P_arranged = {a: [] for a in I}
    AGV_arranged = []
    All_arranged = {}
    precedence_machine = {}
    for a in range(total_Operation):
        All_arranged[key[a]] = []  # Currently arranged operations
        current_machine = key[a][2]  # Machine for the current assignment
        current_operation = key[a][1]  # current assignment
        current_product = key[a][0]  # Current job
        current_op_time = value[a]  # Processing time for the current assignment
        machine_pre = (precedence_machine.get(current_product) or 0)
        if P_arranged[current_product] == []:
            # first transport from LU
            last_op_end_time = TRT[0, current_machine]
        else:
            # the end of previous assignment can be seen as actual finish time + transportation time to next machine
            last_op_end_time = max(P_arranged[current_product])[1] + TRT[machine_pre, current_machine]
        if M_arranged[current_machine] == []:
            ta = max(last_op_end_time, 0)
            arranged(M_arranged, current_machine, P_arranged, current_product, ta, current_op_time, All_arranged,
                     key, a)
            if (TRT[machine_pre, current_machine] != 0):
                # AGV scheduling initial time agv, transportation time, job, machine pre, machine post, initial time next assignment
                AGV_arranged.append(
                    [last_op_end_time - TRT[machine_pre, current_machine], TRT[machine_pre, current_machine],
                     current_product, machine_pre, current_machine, ta])
        else:
            intersection = Find_gap(M_arranged[current_machine])
            inters = copy.deepcopy(intersection)
            while inters:  # Check if it can break out of the loop!
                ta = max(last_op_end_time, inters[0][0])
                if ta + current_op_time <= inters[0][1]:
                    arranged(M_arranged, current_machine, P_arranged, current_product, ta, current_op_time,
                             All_arranged, key, a)
                    if (TRT[machine_pre, current_machine] != 0):
                        AGV_arranged.append(
                            [last_op_end_time - TRT[machine_pre, current_machine], TRT[machine_pre, current_machine],
                             current_product, machine_pre, current_machine, ta])
                    break
                else:
                    inters.pop(0)
        precedence_machine[current_product] = current_machine
        # if last assignment is selected, add agv schedule to report the job to the LU area
        if current_operation == N[current_product]:
            AGV_arranged.append(
                [ta + current_op_time, TRT[current_machine, 0], current_product, current_machine, 0, ta])
    return M_arranged, P_arranged, All_arranged, AGV_arranged

# Do not run separately
def arranged(M_arranged, current_machine, P_arranged, current_product, ta, current_op_time, All_arranged, key, a):
    M_arranged[current_machine] += [(ta, ta + current_op_time)]
    P_arranged[current_product] += [(ta, ta + current_op_time)]
    All_arranged[key[a]] += [ta, ta + current_op_time]
    return M_arranged, P_arranged, All_arranged

# Find the idle time of the machine, do not run separately
def Find_gap(M_arranged):
    arranged = sorted(M_arranged)
    gap_list = []
    if arranged != []:
        for a in range(len(arranged) + 1):
            if a == 0:
                if arranged[a][0] != 0:
                    gap_list.append([0, arranged[a][0]])
            elif a == len(arranged):
                gap_list.append([arranged[a - 1][1], 9999])
            else:
                gap_list.append([arranged[a - 1][1], arranged[a][0]])
    return gap_list

def energy_calculation(ALL_arranged, makespan, AGV_scheduling):
    production_energy=0
    idle_energy = 0
    agv_energy = 0
    aux_energy = 0
    total_production_time_machine = {}
    # production energy
    for key, time_interval in ALL_arranged.items():
        job, operation, machine = key
        start_time, end_time = time_interval
        # Calculate energy consumed for the assignment on that machine during the time interval
        production_energy += PP[(job, operation, machine)]/60 * (end_time - start_time)
        total_production_time_machine[machine] = (total_production_time_machine.get(machine) or 0) + (end_time - start_time)
    # machine in idle
    for machine in M:
        idle_energy += (makespan - (total_production_time_machine.get(machine) or 0)) * IDLE_power[machine-1] /60
    # agv energy
    for task in AGV_scheduling:
        agv_energy += AGV_power/60 * task[1]

    aux_energy = makespan * AUX_power / 60
    # total energy consumed
    tot_energy = production_energy + idle_energy + agv_energy + aux_energy
    return round(tot_energy,2)

O_M_T_total = Decode_OS(Population_ene)
store= []
makespan_tot = []
energy_tot = []
schedule = []
for order in range(len(O_M_T_total)):
    key1 = list(O_M_T_total[order].keys())
    value1 = list(O_M_T_total[order].values())
    schedule_result = Operation_insert(key1, value1)
    makespan_schedule = history[order][1]
    schedule.append(schedule_result)
    # energy calculation
    energy_schedule = energy_calculation(schedule_result[2], makespan_schedule, schedule_result[3])
    store.append([schedule_result, makespan_schedule,energy_schedule])
    makespan_tot.append(makespan_schedule)
    energy_tot.append(energy_schedule)

    if min(energy_tot) >= energy_schedule:
        iteration_energy[order][1] = energy_schedule

select_index, num_pareto = fitness(makespan_tot, energy_tot)
# Find the indexes of the minimum energy values
min_energy_indexes = select_index[:population_number_ene]
DR_energy = []

for idx in range(len(makespan_tot)):
    print(makespan_tot[idx])
for idx in range(len(makespan_tot)):
    print(energy_tot[idx])

print(".")

for idx in min_energy_indexes:
    print(makespan_tot[idx])
for idx in min_energy_indexes:
    print(energy_tot[idx])

for ind in min_energy_indexes:
    print(makespan_tot[ind])
    DR_energy.append(Pop_DR[ind])
for ind in min_energy_indexes:
    print(energy_tot[ind])

final = time.time()
print("Total time", final - initial)

'''
