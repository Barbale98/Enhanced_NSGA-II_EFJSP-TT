import random
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import copy
from Data_reader import TP, TRT, m, n, N, tot_number_operations, M_ij, PP, AGV_power, AUX_power, IDLE_power


total_Operation = tot_number_operations
# Parameters
Pop_size = 100
generation_num = 200
pm = 0.2  # Mutation probability

# Data initialization
M = list(range(1, m + 1))
I = list(range(1, n + 1))
O_ij = {job: list(range(1, N[job] + 1)) for job in range(1, n + 1)}
T_ijm = TP

# Step 1: Encoding and Initialization
class GA():
    def __init__(self, I, M, O_ij, M_ij, T_ijm, Pop_size, total_Operation):
        self.I = I  # Job number
        self.M = M  # Machine number
        self.O_ij = O_ij  # Job_Process
        self.M_ij = M_ij  # Available machines for each assignment
        self.T_ijm = T_ijm  # Processing time for each assignment on available_machine machines
        self.Pop_size = Pop_size  # Population_mks size
        self.total_Operation = total_Operation

    # Random initialization
    def Random_initial(self):
        MS_RS = []
        OS_RS = []

        for i in self.I:
            for j in self.O_ij[i]:
                # Machine part (MS)
                MS_RS.append(random.choice(self.M_ij[i, j]))
                # Operation part (OS)
                OS_RS.append(i)

        random.shuffle(OS_RS)
        RS = OS_RS + MS_RS
        return RS

    ## Second step: Decoding
    # Decoding the machine part from left to right, converting it into a machine order matrix and time order matrix T
    # Decode T = based on machine assignment for each individual get the processing times for the assignment
    def Decode_T(self, Pop_matrix):  # Do not run separately
        T_list = []
        for a in range(len(Pop_matrix)):  # For each chromosome
            T = []
            for b in range(self.total_Operation):
                m = Pop_matrix[:][a][self.total_Operation:self.total_Operation * 2][
                    b]  # Machine for the current assignment
                i_j = list(self.M_ij.keys())[b][0]  # Get the job number for this assignment
                j_i = list(self.M_ij.keys())[b][1]  # Get the assignment number for this assignment
                T_total = self.T_ijm[i_j, j_i, m]
                T.append(T_total)
            T_list.append(T)
        T_matrix = np.array(T_list)
        return T_matrix

    # Decode OS
    def Decode_OS(self, Pop_matrix):  # Decoding
        # The sum of the number of operations of eligiblemachine jobs before the current job
        T_matrix = self.Decode_T(Pop_matrix)
        O_num_list = []
        O_num = 0
        for i in self.I:
            O_num_list.append(O_num)
            O_num += len(self.O_ij[i])

        # Get the corresponding job-assignment group based on the assignment code
        O_M_T_total = []
        for a in range(len(Pop_matrix)):  # For each chromosome
            O_M_T = {}
            for b in range(self.total_Operation):
                O_i = Pop_matrix[:][a][0:self.total_Operation][b]  # OS part of each chromosome
                O_j = list(Pop_matrix[:][a][0:b + 1]).count(
                    O_i)  # The number of times the current sequence number appears, i.e., the assignment number
                T_matrix_column = O_num_list[
                                      O_i - 1] + O_j - 1  # Column number of the current assignment arranged in positive order
                O_M = Pop_matrix[:][a][self.total_Operation:self.total_Operation * 2][
                    T_matrix_column]  # Machine selected for the current assignment
                T_matrix_recent = T_matrix[a, T_matrix_column]  # Time required for the current assignment
                O_M_T[
                    O_i, O_j, O_M] = T_matrix_recent  # Operations sorted by OS code and corresponding equipment fixture
            O_M_T_total.append(O_M_T)
        return O_M_T_total

    # Operation insertion method
    def Operation_insert(self, key, value):
        M_arranged = {a: [] for a in M}
        P_arranged = {a: [] for a in I}
        AGV_arranged = []
        All_arranged = {}
        precedence_machine = {}
        for a in range(self.total_Operation):
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
                self.arranged(M_arranged, current_machine, P_arranged, current_product, ta, current_op_time,
                              All_arranged, key, a)
                if (TRT[machine_pre, current_machine] != 0):
                    # AGV scheduling initial time agv, transportation time, job, machine pre, machine post, initial time next assignment
                    AGV_arranged.append(
                        [last_op_end_time - TRT[machine_pre, current_machine], TRT[machine_pre, current_machine],
                         current_product, machine_pre, current_machine, ta])
            else:
                intersection = self.Find_gap(M_arranged[current_machine])
                inters = copy.deepcopy(intersection)
                while inters:  # Check if it can break out of the loop!
                    ta = max(last_op_end_time, inters[0][0])
                    if ta + current_op_time <= inters[0][1]:
                        self.arranged(M_arranged, current_machine, P_arranged, current_product, ta, current_op_time,
                                      All_arranged, key, a)
                        if (TRT[machine_pre, current_machine] != 0):
                            AGV_arranged.append([last_op_end_time - TRT[machine_pre, current_machine],
                                                 TRT[machine_pre, current_machine], current_product, machine_pre,
                                                 current_machine, ta])
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
    def arranged(self, M_arranged, current_machine, P_arranged, current_product, ta, current_op_time, All_arranged, key,
                 a):
        M_arranged[current_machine] += [(ta, ta + current_op_time)]
        P_arranged[current_product] += [(ta, ta + current_op_time)]
        All_arranged[key[a]] += [ta, ta + current_op_time]
        return M_arranged, P_arranged, All_arranged

    # Find the idle time of the machine, do not run separately
    def Find_gap(self, M_arranged):
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

    # Calculate the fitness of each individual
    def makespan_calculation(self, time):
        time_values = []
        # add transportation time to LU to calculate makespan
        # just check the time of last operations
        for key in time.keys():
            job, operation, machine = key
            if operation == N[job]:
                time_values.append(time[key][1] + TRT[(machine, 0)])
        return max(time_values)

    def energy_calculation(self, ALL_arranged, makespan, AGV_scheduling):
        production_energy = 0
        idle_energy = 0
        agv_energy = 0
        aux_energy = 0
        total_production_time_machine = {}
        # production energy
        for key, time_interval in ALL_arranged.items():
            job, operation, machine = key
            start_time, end_time = time_interval
            # Calculate energy consumed for the assignment on that machine during the time interval
            production_energy += PP[(job, operation, machine)] / 60 * (end_time - start_time)
            total_production_time_machine[machine] = (total_production_time_machine.get(machine) or 0) + (
                    end_time - start_time)
        # machine in idle
        for machine in M:
            idle_energy += (makespan - (total_production_time_machine.get(machine) or 0)) * IDLE_power[machine - 1] / 60
        # agv energy
        for task in AGV_scheduling:
            agv_energy += AGV_power / 60 * task[1]
        aux_energy = makespan * AUX_power / 60
        # total energy consumed
        tot_energy = production_energy + idle_energy + agv_energy + aux_energy
        return round(tot_energy, 4)

    def crowding_distance_sort(self, last_front):
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

    def fast_non_dominated_sort(self, combined_results):
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
                    if self.check_dominance(individual1, individual2):
                        dominated_solutions[index1].add(index2)
                    elif self.check_dominance(individual2, individual1):
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

    def fitness_VNS(self, time1, energy1):
        combined_results = []
        for i in range(len(time1)):
            combined_results.append([time1[i], energy1[i]])
        fronts = self.fast_non_dominated_sort(combined_results)

        selected_individuals = []
        current_front = 0

        Pareto_ind = len(list(fronts[0]))

        while len(selected_individuals) < len(time1):
                selected_individuals.extend(list(fronts[current_front]))
                current_front += 1

        return selected_individuals, Pareto_ind

    def fitness(self, time1, energy1):
        combined_results = []
        for i in range(len(time1)):
            combined_results.append([time1[i], energy1[i]])
        fronts = self.fast_non_dominated_sort(combined_results)

        selected_individuals = []
        current_front = 0

        Pareto_ind = len(list(fronts[0]))

        while len(selected_individuals) < min(self.Pop_size, len(time1)):

            if len(fronts[current_front]) + len(selected_individuals) <= min(self.Pop_size, len(time1)):
                selected_individuals.extend(list(fronts[current_front]))
                current_front += 1
            else:
                # Sort the last Pareto front based on crowding distance
                keys_last_front = list(fronts[current_front])

                # Use keys_last_front to filter the combined_results list
                last_front_individuals = [combined_results[key] for key in keys_last_front]
                distances = self.crowding_distance_sort(last_front_individuals)
                sorted_last_front = [ind for _, ind in
                                     sorted(zip(distances, keys_last_front), key=lambda x: x[0], reverse=True)]

                selected_individuals.extend(sorted_last_front[:(min(self.Pop_size, len(time1)) - len(selected_individuals))])
                # rank individuals on the front based on crowding distance and peak the best ones

        return selected_individuals, Pareto_ind

    # for mating pool selection
    def tournament_selection(self, time1, energy1, size_matingpool):
        combined_results = []
        for i in range(len(time1)):
            combined_results.append([time1[i], energy1[i]])
        fronts = self.fast_non_dominated_sort(combined_results)
        sorted_front = []
        selected_mating_pool = []
        # sort each front based on crowding distance

        for front in fronts:
            if front != set():
                # Sort the last Pareto front based on crowding distance
                keys_front = list(front)
                # Use keys_last_front to filter the combined_results list
                front_individuals = [combined_results[key] for key in keys_front]

                distances = self.crowding_distance_sort(front_individuals)

                sorted_front.append(
                    [ind for _, ind in sorted(zip(distances, keys_front), key=lambda x: x[0], reverse=True)])
        count = 0
        while len(selected_mating_pool) < size_matingpool:
            # Randomly select two distinct indices
            i1, i2 = random.sample(range(len(combined_results)), 2)

            # Determine the front index and position for each selected individual
            front_index_i1, position_i1 = next(
                ((index, front.index(i1)) for index, front in enumerate(sorted_front) if i1 in front), (None, None))
            front_index_i2, position_i2 = next(
                ((index, front.index(i2)) for index, front in enumerate(sorted_front) if i2 in front), (None, None))

            # Compare the fronts and select the index from the lower front
            if front_index_i1 < front_index_i2:
                winner = i1
            elif front_index_i2 < front_index_i1:
                winner = i2
            else:
                # If in the same front, select the one that appears first, leveraging crowding distance
                winner = i1 if position_i1 < position_i2 else i2

            # Add the winner to the mating pool if it's not going to mate to himself
            if len(selected_mating_pool) < int(Pop_size / 2):
                selected_mating_pool.append(winner)
            else:
                if winner != selected_mating_pool[count]:
                    count += 1
                    selected_mating_pool.append(winner)

        return selected_mating_pool
    def check_dominance(self, solution1, solution2):
        """
        - bool: True if solution1 dominates solution2, False otherwise.
        """
        dominates = all(s1 <= s2 for s1, s2 in zip(solution1, solution2)) and any(
            s1 < s2 for s1, s2 in zip(solution1, solution2))
        return dominates

    # Crossover, OS part (IPOX)
    def IPOX(self, p1_OS, p2_OS):
        num = random.randint(1, len(self.I) - 2)  # Choose a random number
        set1 = random.sample(self.I, k=num)  # num jobs are placed in set1
        c1_OS = np.zeros(self.total_Operation, dtype=int)  # Initialize offspring
        c2_OS = np.zeros(self.total_Operation, dtype=int)
        c2_left = []
        c1_left = []
        for a in range(
                len(p1_OS)):  # The first parent chromosome has only jobs belonging to set1, in c1 indexed, in c2 in order
            if p1_OS[a] in set1:
                c1_OS[a] = p1_OS[a]
            else:
                c1_left.append(p1_OS[a])
            if p2_OS[a] in set1:
                c2_OS[a] = p2_OS[a]
            else:
                c2_left.append(p2_OS[a])
        idx1 = -1
        idx2 = -1
        for c in range(self.total_Operation):
            if c1_OS[c] == 0:  # If this position is 0, it does not belong to set1
                c1_OS[c] = c2_left[idx1]  # Reverse order
                idx1 -= 1
            if c2_OS[c] == 0:
                c2_OS[c] = c1_left[idx2]
                idx2 -= 1
        return c1_OS, c2_OS

    # Uniform crossover, MS
    def UX(self, p1, p2):
        index = []
        num = random.randint(1, self.total_Operation)
        for a in range(0, self.total_Operation):
            index.append(a)
        set1 = random.sample(index, k=num)
        set2 = list(set(index).difference(set(set1)))
        c1_MS = np.zeros(self.total_Operation, dtype=int)
        c2_MS = np.zeros(self.total_Operation, dtype=int)
        for a in set1:
            c1_MS[a] = p1[a]
            c2_MS[a] = p2[a]
        for b in set2:
            c1_MS[b] = p2[b]
            c2_MS[b] = p1[b]
        return c1_MS, c2_MS

    # OS mutation (swap any two positions)
    def swap_mutation(self, os):
        index = []
        for a in range(0, self.total_Operation):
            index.append(a)
        set = random.sample(index, k=2)
        temp = os[set[0]]
        os[set[0]] = os[set[1]]
        os[set[1]] = temp
        return os

    # MS mutation, reassign random operation to the best PT or best EC
    def Random_MS(self, ms):

        idx = random.randint(0, self.total_Operation - 1)
        i_j = list(self.M_ij.keys())[idx][0]  # Get the job number of the current assignment
        j_i = list(self.M_ij.keys())[idx][1]  # Get the assignment number of the current assignment

        rand = random.random()
        if rand < 0.5:
            machine_time = []
            for machine_idx in self.M_ij[i_j, j_i]:
                machine_time.append(self.T_ijm[i_j, j_i, machine_idx])
            min_value = min(machine_time)
            min_indexes = [index for index, value in enumerate(machine_time) if value == min_value]
            new_idx = random.choice(min_indexes)
        else:
            machine_energy = []
            for machine_idx in self.M_ij[i_j, j_i]:
                machine_energy.append(round(TP[i_j, j_i, machine_idx] / 60 * PP[i_j, j_i, machine_idx], 1))
            min_value = min(machine_energy)
            min_indexes = [index for index, value in enumerate(machine_energy) if value == min_value]
            new_idx = random.choice(min_indexes)

        ms[idx] = self.M_ij[i_j, j_i][new_idx]
        return ms

    def swap_os(self, os, job1, operation1, job2, operation2):
        occurrence_count = 0
        # Loop through the jobs_vector to find the index of the desired occurrence
        for index, job in enumerate(os):
            if job == job1:
                occurrence_count += 1
                if occurrence_count == operation1:
                    index1 = index
                    break  # Stop searching once we find the desired occurrence
        occurrence_count = 0
        # Loop through the jobs_vector to find the index of the desired occurrence
        for index, job in enumerate(os):
            if job == job2:
                occurrence_count += 1
                if occurrence_count == operation2:
                    index2 = index
                    break  # Stop searching once we find the desired occurrence
        temp = os[index1]
        os[index1] = os[index2]
        os[index2] = temp
        return os

    ## OS mutation (swap an assignment of the last 2 jobs that end  trying to anticipate them)
    # Critic assignment may be one that makes wait for precedence assignment or last assignment
    def smart_swap_mutation(self, cos, schedule):
        cos = []
        for job in schedule[2]:
            cos.append(job[0])

        job_critic = max(schedule[1], key=lambda x: schedule[1][x][-1][1])
        random_operation = random.choice([1, len(O_ij[job_critic])])

        # select the correct assignment to swap
        for key in schedule[2].keys():
            if key[0] == job_critic and key[1] == random_operation:
                machine_critic = key[2]  # Return the machine part of the key
        if random_operation == 1 and schedule[2][(job_critic, random_operation, machine_critic)][0] == TRT[
            0, machine_critic]:
            random_operation = random.choice([2, len(O_ij[job_critic])])
            for key in schedule[2].keys():
                if key[0] == job_critic and key[1] == random_operation:
                    machine_critic = key[2]  # Return the machine part of the key
        count = 0
        var = True
        while var is True:
            if random_operation in [2, len(O_ij[job_critic])]:
                # Find the key for the previous assignment
                previous_key = next(((job, op, machine) for (job, op, machine), _ in schedule[2].items() if
                                     job == job_critic and op == random_operation - 1), None)
                if previous_key is not None:
                    if schedule[2][(job_critic, random_operation, machine_critic)][0] == schedule[2][previous_key][1] + TRT[previous_key[2], machine_critic]:
                        random_operation = random.choice([2, len(O_ij[job_critic])])
                        for key in schedule[2].keys():
                            if key[0] == job_critic and key[1] == random_operation:
                                machine_critic = key[2]  # Return the machine part of the key
                        count += 1
                        if count == 10:
                            var = False
                    else:
                        var = False
                else:
                    var = False
            else:
                var = False

        job_critic_key = (job_critic, random_operation, machine_critic)
        job_critic_time_frame = schedule[2].get(job_critic_key)
        # Filter tasks performed on machine_critic and before the end time of job_critic's assignment
        filtered_tasks = [(job, op, machine, start_end) for (job, op, machine), start_end in schedule[2].items()
                          if machine == machine_critic and start_end[1] <= job_critic_time_frame[
                              0] and job != job_critic]

        # Sort filtered tasks by their end time to find the task closest to the start of job_critic's assignment
        filtered_tasks.sort(key=lambda x: x[3][1], reverse=True)  # Sort in descending order by end time

        # Extract the job and assignment number of the task closest to job_critic's assignment start time
        if filtered_tasks:
            # Parameters for the job to find and its occurrence [(4, 2, 8, [40, 64]), (5, 2, 8, [22, 40]), (14, 1, 8, [2, 22])]
            closest_task = filtered_tasks[0]  # The first item after sorting will be the closest
            job_to_find = closest_task[0]  # Example job number to find
            occurrence_to_find = closest_task[1]  # Example occurrence (1st, 2nd, etc.)

            cos = Pop.swap_os(cos, job_to_find, occurrence_to_find, job_critic, random_operation)
        return cos

    # OS mutation (identifies and swap a job in a critical block)
    def os_critical_block_mutation(self, cos, schedule):
        cos = []
        for job in schedule[2]:
            cos.append(job[0])
        list_machines = copy.deepcopy(M)
        random.shuffle(list_machines)

        # Sort the schedule based on the machine's index in list_machines, then by end time
        sorted_schedule = sorted(schedule[2].items(), key=lambda x: (list_machines.index(x[0][2]), x[1][1]))
        # Convert the sorted list of tuples back to a dictionary if needed
        sorted_schedule_dict = dict(sorted_schedule)

        block_operations = {}
        current_end = []
        for assignment in sorted_schedule_dict:
            machine = assignment[2]

            if machine not in block_operations:
                current_end.append(-1)
                block_operations[machine] = []

            if current_end[-1] == sorted_schedule_dict[assignment][0]:
                block_operations[machine][-1].append(assignment)
            else:
                block_operations[machine].append([assignment])
            current_end.append(sorted_schedule_dict[assignment][1])

        # random machines in which create the swap
        machine_choices = [machine for machine in schedule[0] if schedule[0][machine]]
        random_machines = random.sample(machine_choices, k=1)

        # random gaps for each selected machines
        random_indexes = {}
        for machine in random_machines:
            block_len = len(block_operations[machine])
            if block_len > 1:
                random_indexes[machine] = random.sample(range(0, block_len), k=1)
            else:
                random_indexes[machine] = [0]  # Default to the only block if there's only one
        # some indexes get swaped inside the block
        # the eligible machine gets the previous operation that causes the waiting get swapped before
        for machine in random_machines:
            for index, gap in enumerate(block_operations[machine]):
                if index in random_indexes[machine]:
                    if len(gap) > 1:
                        # If first block, swap first 2 operations of the block only if first operation starts after time required to reach the machine
                        if index == len(block_operations[machine]):
                            job1 = gap[0][0]
                            operation1 = gap[0][1]
                            job2 = gap[1][0]
                            operation2 = gap[1][1]
                            cos = Pop.swap_os(cos, job1, operation1, job2, operation2)
                        elif index == 0:
                            job1 = gap[-1][0]
                            operation1 = gap[-1][1]
                            job2 = gap[-2][0]
                            operation2 = gap[-2][1]
                            cos = Pop.swap_os(cos, job1, operation1, job2, operation2)
                        # If last block, swap first 2 operations of the block
                        elif index != 0 and index != len(block_operations[machine]):
                            job1 = gap[0][0]
                            operation1 = gap[0][1]
                            job2 = gap[1][0]
                            operation2 = gap[1][1]
                            cos = Pop.swap_os(cos, job1, operation1, job2, operation2)
                            if len(gap) > 2:
                                job1 = gap[-1][0]
                                operation1 = gap[-1][1]
                                job2 = gap[-2][0]
                                operation2 = gap[-2][1]
                                cos = Pop.swap_os(cos, job1, operation1, job2, operation2)
        return cos

    # MS mutation for VNS
    def smart_MS_1(self, cms, schedule):
        # reassign a random operation to its less utilised eligible machine
        idx = random.randint(0, self.total_Operation - 1)
        i_j = list(self.M_ij.keys())[idx][0]  # Get the job number of the current assignment
        j_i = list(self.M_ij.keys())[idx][1]  # Get the assignment number of the current assignment
        # Calculate total assignment time for each machine
        machine_operation_time = {machine: sum(end - start for start, end in times) for machine, times in
                                  schedule[0].items() if machine in M_ij[i_j, j_i]}
        # Sort the machines based on total assignment time
        sorted_machines_by_utilization = sorted(machine_operation_time.items(), key=lambda item: item[1])
        new_idx = sorted_machines_by_utilization[0][0]  # Get the machine with the minimum utilization time
        cms[idx] = self.M_ij[i_j, j_i][M_ij[i_j, j_i].index(new_idx)]
        return cms

    def smart_MS_2(self, cms, schedule):
        # reassign an operation performed on the machine that finishes last
        filtered_schedule = {key: value for key, value in schedule[0].items() if value and len(value[-1]) > 1}
        sorted_machines = sorted(filtered_schedule, key=lambda x: filtered_schedule[x][-1][1], reverse=True)
        critic_machine = sorted_machines[0]
        # Randomly choose a (job, assignment) performed on critic_machine
        filtered_entries = [(job, operation) for (job, operation, machine), times in schedule[2].items() if
                            machine == critic_machine]
        random_job_operation = random.choice(filtered_entries)
        idx = -1
        for el in N:
            if el != random_job_operation[0]:
                idx += N[el]
            else:
                idx += random_job_operation[1]
                break
        i_j = list(self.M_ij.keys())[idx][0]  # Get the job number of the current assignment
        j_i = list(self.M_ij.keys())[idx][1]  # Get the assignment number of the current assignment
        # Filter out the critic_machine from the list of machines for the given job and operation
        machines_except_critic = [machine for machine in self.M_ij[i_j, j_i] if machine != critic_machine]
        # If there are machines left after excluding the critic_machine, choose one at random
        if machines_except_critic:
            new_idx = random.choice(machines_except_critic)
            cms[idx] = new_idx
        return cms

    def smart_MS_3(self, cms, schedule):
        # Worst assignments for energy is reassigned to the best
        filtered_diff = {key: diff_dict[key] for key in diff_dict if key in schedule[2]}
        max_key = max(filtered_diff, key=filtered_diff.get)
        i_j, j_i, idx = max_key[0], max_key[1], list(filtered_diff.keys()).index(
            max_key)  # Extracting job and operation
        machine_energy = []
        for machine_idx in self.M_ij[i_j, j_i]:
            machine_energy.append(energy[i_j, j_i, machine_idx])
        min_value = min(machine_energy)
        min_indexes = [index for index, value in enumerate(machine_energy) if value == min_value]
        new_idx = random.choice(min_indexes)
        cms[idx] = self.M_ij[i_j, j_i][new_idx]
        return cms

    def smart_MS_4(self, cms, schedule):
        # Worst assignments for time is reassigned to the best
        filtered_diff = {key: diff_best[key] for key in diff_best if key in schedule[2]}
        max_key = max(filtered_diff, key=filtered_diff.get)
        i_j, j_i, idx = max_key[0], max_key[1], list(filtered_diff.keys()).index(
            max_key)  # Extracting job and operation
        machine_time = []
        for machine_idx in self.M_ij[i_j, j_i]:
            machine_time.append(self.T_ijm[i_j, j_i, machine_idx])
        min_value = min(machine_time)
        min_indexes = [index for index, value in enumerate(machine_time) if value == min_value]
        new_idx = random.choice(min_indexes)
        cms[idx] = self.M_ij[i_j, j_i][new_idx]
        return cms


def gantt(result_sch):
    # ALL contains the (job,assignment,machine): [initial time,final time]
    ALL = result_sch[2]
    fig, ax = plt.subplots()
    makespan = 0
    # colors
    unique_job_ids = set(range(1, n + 1))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_job_ids)))

    product_colors = {}  # Dictionary to store product ID-color mapping

    for jobs, color in zip(unique_job_ids, colors):
        product_colors[jobs] = color

    for key in ALL.keys():
        color = product_colors[key[0]]
        ax.barh(key[2], width=ALL[key][1] - ALL[key][0], height=0.6, left=ALL[key][0], color=color, edgecolor='black',
                linewidth=0.3)
        ax.text(ALL[key][0] + (ALL[key][1] - ALL[key][0]) / 2, key[2], str(key[0]) + "," + str(key[1]), ha='center',
                va='center', fontsize=8)
        if ALL[key][1] > makespan:
            makespan = ALL[key][1]

    for i, (t_inizio, duration, prodotto, mac_pre, mac_post, ta) in enumerate(result_sch[3]):
        if mac_pre != 0 and mac_post != 0:
            ax.barh(mac_pre + 0.4, duration, left=t_inizio, height=0.2, color='orange',
                    edgecolor='black')
            ax.text(t_inizio + duration / 2, mac_pre + 0.4, str(prodotto) + str(mac_pre) + str(mac_post), ha='center',
                    va='center', color='black', fontsize=6)
        if mac_pre == 0:
            ax.barh(mac_post - 0.4, duration, left=ta - duration, height=0.2, color='orange',
                    edgecolor='black')
            ax.text(ta - duration / 2, mac_post - 0.4, str(prodotto) + 'LU' + str(mac_post), ha='center',
                    va='center', color='black', fontsize=6)
        if mac_post == 0:
            if duration != 0:
                ax.barh(mac_pre - 0.4, duration, left=t_inizio, height=0.2, color='orange',
                        edgecolor='black')
                ax.text(t_inizio + duration / 2, mac_pre - 0.4, str(prodotto) + str(mac_pre) + 'LU', ha='center',
                        va='center', color='black', fontsize=6)

    # Determine the locator parameters based on the makespan
    if makespan <= 100:
        major_tick_locator = 5
        minor_tick_locator = 1
    elif makespan <= 200:  # Adjust these ranges as needed
        major_tick_locator = 10
        minor_tick_locator = 5
    elif makespan <= 400:  # Adjust these ranges as needed
        major_tick_locator = 25
        minor_tick_locator = 5
    else:
        major_tick_locator = 50
        minor_tick_locator = 10
    # Set the locator for the major ticks
    ax.xaxis.set_major_locator(ticker.MultipleLocator(major_tick_locator))
    # For minor ticks, set them according to the determined interval
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_tick_locator))
    # Only draw grid lines for the minor ticks (which are at every single unit)
    ax.grid(which='minor', axis='x', linestyle=':', alpha=0.1)
    # Optionally, if you want to see major grid lines as well (at multiples of 5), you can enable this:
    ax.grid(which='major', axis='x', linestyle=':', alpha=0.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_yticks(range(1, m + 1))
    ax.set_yticklabels([f"M{i}" for i in range(1, m + 1)])

    plt.show()

def pareto_front(gen_result):
    plt.figure()

    makespans = []
    Energy = []
    individuals_dominate = {i: 0 for i in range(len(gen_result))}

    for index1, individual1 in enumerate(gen_result):
        for index2, individual2 in enumerate(gen_result):
            if Pop.check_dominance(individual1, individual2):
                individuals_dominate[index2] += 1

    for i, individual in enumerate(gen_result):
        if individuals_dominate[i] == 0:
            makespans.append(individual[0])
            Energy.append(individual[1])

    # Plotting Pareto front for the current generation with colors
    plt.scatter(makespans, Energy, marker='o', c='orange')

    # Combine makespans and energy into a list of tuples
    combined = list(zip(makespans, Energy))

    # Convert the list of tuples into a set to remove duplicates
    unique_combinations = set(combined)
    print("Pareto Front:")
    for makespan, energy in sorted(unique_combinations):  # Sorting for better readability
        print(makespan)
    for makespan, energy in sorted(unique_combinations):  # Sorting for better readability
        print(energy)

    plt.xlabel('Makespan')
    plt.ylabel('Energy')
    plt.title(f'Pareto Fronts Solution')
    plt.show()

    return sorted(unique_combinations)

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

# energy consumed by a set of (job,assignment,machine) is given as PROD. TIME * PROD.POWER + PAUX * PROC. TIME / N_MACHINES
energy = {}
for element in [x for x in TP.keys()]:
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
        diff_dict[(job_op[0], job_op[1], machine)] = round(current_energy - best_energy,1)

old_time = time.time()
Pop = GA(I, M, O_ij, M_ij, T_ijm, Pop_size, total_Operation)


# Generating initial population
gen = 0
result_time_list = []
result_energy_list = []
result_time_list_energy = []
result_energy_list_makespan = []
draw_result_makespan = []
draw_result_energy = []
store_result = []
data_gen_store = []
pop_store_results = []
vns_total = []
iteration = []
iteration_energy = []

num_pareto = 0
for gennum in range(generation_num):
    gen += 1
    if gennum == 0:
        # randomly generate initial population
        # store individuals in pop list
        Pop_list = []
        for i in range(Pop_size):
            Pop_list.append(Pop.Random_initial())

        Pop_matrix = np.array(Pop_list)

        # Operation Insertion Method
        O_M_T_total = Pop.Decode_OS(Pop_matrix)

        schedule_result_total = []
        makespan_total = []
        energy_total = []
        data_gen = []
        # For each chromosome, find the fitness and minimum value of the entire population
        for order in range(len(O_M_T_total)):
            key1 = list(O_M_T_total[order].keys())
            value1 = list(O_M_T_total[order].values())
            schedule_result = Pop.Operation_insert(key1, value1)
            schedule_result_total.append(schedule_result)  # Decoding results of the population
            makespan_schedule = Pop.makespan_calculation(schedule_result[2])
            makespan_total.append(makespan_schedule)
            # energy calculation
            energy_schedule = Pop.energy_calculation(schedule_result[2], makespan_schedule, schedule_result[3])
            energy_total.append(energy_schedule)
            store_result.append([makespan_schedule, energy_schedule])
            data_gen.append([makespan_schedule, energy_schedule])

        pop_store_results.append(data_gen)
        data_gen_store.append(data_gen)

    else:
        data_gen = []
        # Generating a new population
        Pop_total = np.vstack((Pop_matrix, son_pop_matrix))

        # Operation Insertion Method
        O_M_T_total = Pop.Decode_OS(Pop_total)
        schedule_result_total_else = []
        makespan_total_else = []
        energy_total_else = []

        for order in range(len(O_M_T_total)):  # For each chromosome, find the minimum value of the entire population
            key1 = list(O_M_T_total[order].keys())
            value1 = list(O_M_T_total[order].values())
            schedule_result = Pop.Operation_insert(key1, value1)
            schedule_result_total_else.append(schedule_result)  # Decoding results of the population
            makespan_schedule = Pop.makespan_calculation(schedule_result[2])
            makespan_total_else.append(makespan_schedule)
            # energy calculation
            energy_schedule = Pop.energy_calculation(schedule_result[2], makespan_schedule, schedule_result[3])
            energy_total_else.append(energy_schedule)
            store_result.append([makespan_schedule, energy_schedule])
            data_gen.append([makespan_schedule, energy_schedule])

        data_gen_store.append(data_gen)

        # add fitness function
        # I need to return for each individual of the merged populations its rank and its crowding distance
        select_index, num_pareto = Pop.fitness(makespan_total_else, energy_total_else)
        print(num_pareto)
        Pop_list = []
        schedule_result_total = []
        makespan_total = []
        energy_total = []
        pop_results = []
        for a in select_index:
            Pop_list.append(Pop_total[a])
            makespan_total.append(makespan_total_else[a])
            energy_total.append(energy_total_else[a])
            schedule_result_total.append(schedule_result_total_else[a])
            pop_results.append([makespan_total_else[a], energy_total_else[a]])
        pop_store_results.append(pop_results)
        Pop_matrix = np.array(Pop_list)

    ##index
    index_pop = makespan_total.index(min(makespan_total))
    result_cho = Pop_matrix[index_pop]
    result_sch = schedule_result_total[index_pop]

    # Calculate and print average, max, and min makespan
    avg_makespan = round(np.mean(makespan_total), 2)
    max_makespan = max(makespan_total)
    min_makespan = min(makespan_total)
    avg_energy = round(np.mean(energy_total), 2)
    max_energy = round(max(energy_total), 2)
    min_energy = round(min(energy_total), 2)
    iteration.append([time.time() - old_time, min_makespan])
    iteration_energy.append([time.time() - old_time, min_energy])
    print("*")
    print('Gen:', gen)
    print('Makespan: Min:', min_makespan, 'Max:', max_makespan, 'Average:', avg_makespan)
    print('Energy: Min:', min_energy, 'Max:', max_energy, 'Average:', avg_energy)
    print('Best Makespan:', min_makespan, "Best makespan energy:",
          energy_total[makespan_total.index(min(makespan_total))])
    print('Best Energy:', min_energy,
          'Best energy makespan:', makespan_total[energy_total.index(min(energy_total))])

    # Crossover
    C_pop_total = []

    # APPLY TOURNAMENT SELECTION FOR MATING POOL
    new_index = Pop.tournament_selection(makespan_total, energy_total, Pop_size)

    for a in range(int(len(new_index) / 2)):
        # Generate combinations of parents, perform crossover once in each loop, and generate the same number of offspring as parents
        p1_idx = new_index[a]
        p2_idx = new_index[int(len(new_index) / 2) + a]
        p1_OS = list(Pop_matrix[p1_idx, :][0:total_Operation])  # Extract the OS segment from the initial population
        p2_OS = list(Pop_matrix[p2_idx, :][0:total_Operation])
        p1_MS = list(Pop_matrix[p1_idx, :][total_Operation:total_Operation * 2])
        p2_MS = list(Pop_matrix[p2_idx, :][total_Operation:total_Operation * 2])
        c1_OS, c2_OS = Pop.IPOX(p1_OS, p2_OS)
        c1_MS, c2_MS = Pop.UX(p1_MS, p2_MS)
        C_pop_total.append(list(c1_OS) + list(c1_MS))
        C_pop_total.append(list(c2_OS) + list(c2_MS))

    ## Mutation
    son_pop_total = []  # Mutation result
    for list_pop in C_pop_total:
        if random.random() < pm:
            os = list_pop[0:total_Operation]
            ms = list_pop[total_Operation:total_Operation * 2]
            c_os = Pop.swap_mutation(os)
            c_ms = Pop.Random_MS(ms)
            son_pop_total.append(c_os + c_ms)
        else:
            son_pop_total.append(list_pop)

    # Remove identical schedules
    if gen in range(49, generation_num, 50):
        duplicates = set()
        # Compare every pair of selected indexes
        for i, a1 in enumerate(select_index):
            for a2 in select_index[i + 1:]:  # Start from the next element to avoid comparing the same pair twice
                # do not remove more than the number of sons to avoid errors of next population selection
                if schedule_result_total_else[a1][2] == schedule_result_total_else[a2][2] and len(duplicates) < len(
                        son_pop_total):
                    duplicates.add(a2)
        print(duplicates)
        select_unique_index = [index for index in select_index if index not in duplicates]
        Pop_list = []
        for a in select_unique_index:
            Pop_list.append(Pop_total[a])

        # Select parents for Variable Neighbourhood Search every 5 generations
    if gen in range(39, generation_num, 20):

        pareto_individuals = random.sample(range(min(num_pareto, 100)), k=min(num_pareto, 2))
        VNS_individuals = copy.deepcopy(pareto_individuals)
        Other_individuals = random.sample(
            [ind for ind in range(len(Pop_matrix) - 1) if ind not in pareto_individuals],
            k=(20 - len(pareto_individuals)))
        VNS_individuals.extend(Other_individuals)

        for a in VNS_individuals:
            for i in range(10):

                if i == 0:
                    local = []

                    list_pop = list(Pop_matrix[a])

                    # add the parent to the local search
                    local.append(list_pop)
                    # remove parent from population
                    Pop_list = [sublist for sublist in Pop_list if list(sublist) != list_pop]
                    os_copy = list_pop[0:total_Operation]
                    ms_copy = list_pop[total_Operation:total_Operation * 2]

                    os = list_pop[0:total_Operation]
                    c_os = Pop.smart_swap_mutation(os, schedule_result_total_else[a])
                    c_ms = list_pop[total_Operation:total_Operation * 2]
                    if c_os != os_copy:
                        local.append(c_os + c_ms)

                    os = copy.deepcopy(os_copy)
                    c_os = Pop.os_critical_block_mutation(os, schedule_result_total_else[a])
                    c_ms = list_pop[total_Operation:total_Operation * 2]
                    if c_os != os_copy:
                        local.append(c_os + c_ms)

                    ms = list_pop[total_Operation:total_Operation * 2]
                    c_os = list_pop[0:total_Operation]
                    c_ms = Pop.smart_MS_1(ms, schedule_result_total_else[a])
                    if c_ms != ms_copy:
                        local.append(c_os + c_ms)

                    ms = list_pop[total_Operation:total_Operation * 2]
                    c_os = list_pop[0:total_Operation]
                    c_ms = Pop.smart_MS_2(ms, schedule_result_total_else[a])
                    if c_ms != ms_copy:
                        local.append(c_os + c_ms)

                    ms = list_pop[total_Operation:total_Operation * 2]
                    c_os = list_pop[0:total_Operation]
                    c_ms = Pop.smart_MS_3(ms, schedule_result_total_else[a])
                    if c_ms != ms_copy:
                        local.append(c_os + c_ms)

                    ms = list_pop[total_Operation:total_Operation * 2]
                    c_os = list_pop[0:total_Operation]
                    c_ms = Pop.smart_MS_4(ms, schedule_result_total_else[a])
                    if c_ms != ms_copy:
                        local.append(c_os + c_ms)

                else:
                    local = []
                    for index in indexes:
                        os = local_copy[index][0:total_Operation]
                        ms = local_copy[index][total_Operation:total_Operation * 2]
                        c_os = Pop.smart_swap_mutation(os, schedule_result_total_else_local[index])
                        c_ms = ms
                        if c_os != local_copy[index][0:total_Operation]:
                            local.append(c_os + c_ms)

                        os = local_copy[index][0:total_Operation]
                        ms = local_copy[index][total_Operation:total_Operation * 2]
                        c_os = Pop.os_critical_block_mutation(os, schedule_result_total_else_local[index])
                        c_ms = ms
                        if c_os != local_copy[index][0:total_Operation]:
                            local.append(c_os + c_ms)

                        ms = local_copy[index][total_Operation:total_Operation * 2]
                        c_os = local_copy[index][0:total_Operation]
                        c_ms = Pop.smart_MS_1(ms, schedule_result_total_else[a])
                        if c_ms != local_copy[index][total_Operation:total_Operation * 2]:
                            local.append(c_os + c_ms)

                        ms = local_copy[index][total_Operation:total_Operation * 2]
                        c_os = local_copy[index][0:total_Operation]
                        c_ms = Pop.smart_MS_2(ms, schedule_result_total_else[a])
                        if c_ms != local_copy[index][total_Operation:total_Operation * 2]:
                            local.append(c_os + c_ms)

                        ms = local_copy[index][total_Operation:total_Operation * 2]
                        c_os = local_copy[index][0:total_Operation]
                        c_ms = Pop.smart_MS_3(ms, schedule_result_total_else[a])
                        if c_ms != local_copy[index][total_Operation:total_Operation * 2]:
                            local.append(c_os + c_ms)

                        ms = local_copy[index][total_Operation:total_Operation * 2]
                        c_os = local_copy[index][0:total_Operation]
                        c_ms = Pop.smart_MS_4(ms, schedule_result_total_else[a])
                        if c_ms != local_copy[index][total_Operation:total_Operation * 2]:
                            local.append(c_os + c_ms)

                        local.append(local_copy[index])

                # get the pareto front of the neighbour
                Neighbour = np.array(local)
                O_M_T_total = Pop.Decode_OS(Neighbour)
                schedule_result_total_else_local = []
                makespan_total_else_local = []
                energy_total_else_local = []
                for order in range(len(O_M_T_total)):
                    key1 = list(O_M_T_total[order].keys())
                    value1 = list(O_M_T_total[order].values())
                    schedule_result = Pop.Operation_insert(key1, value1)
                    schedule_result_total_else_local.append(schedule_result)
                    makespan_schedule = Pop.makespan_calculation(schedule_result[2])
                    energy_schedule = Pop.energy_calculation(schedule_result[2], makespan_schedule,
                                                             schedule_result[3])
                    makespan_total_else_local.append(makespan_schedule)
                    energy_total_else_local.append(energy_schedule)

                select_index_local, num_pareto_local = Pop.fitness_VNS(makespan_total_else_local,
                                                                   energy_total_else_local)
                if min(energy_total_else_local) < min_energy:
                    min_energy = min(energy_total_else_local)
                    iteration_energy.append([time.time() - old_time, min_energy])
                if min(makespan_total_else_local) < min_makespan:
                    min_makespan = min(makespan_total_else_local)
                    iteration.append([time.time() - old_time, min_makespan])

                # Local selection
                it = 0

                taken = []
                indexes = []
                for a1 in select_index_local:
                    if it == max(num_pareto_local, 10) and a not in pareto_individuals:
                        break
                    if it == max(num_pareto_local, 40) and a in pareto_individuals:
                        break
                    if it == num_pareto_local and i == 9:
                        break

                    if [makespan_total_else_local[a1], energy_total_else_local[a1]] not in taken:
                        indexes.append(a1)
                        taken.append([makespan_total_else_local[a1], energy_total_else_local[a1]])

                    it += 1

                local_copy = copy.deepcopy(local)

            for index in indexes:
                Pop_list.append(Neighbour[index])

        Pop_matrix = np.array(Pop_list)

    son_pop_matrix = np.array(son_pop_total)  # Mutation result

    current_time = time.time()
    print("The running time is " + str(round(current_time - old_time, 2)) + "s")

    '''
    makespans = []
    final_energy = []
    colors = []  # This will store the color (population index) for each individual

    # Extract makespans, energy values, and assign colors for all individuals in all populations
    for pop_index, population in enumerate(pop_store_results):
        for individual in population:
            makespans.append(individual[0])  # Makespan
            final_energy.append(individual[1])     # Energy

    colors = np.arange(len(pop_store_results))
    # Repeat each element in the colors vector by the number of individuals in each population
    colors = np.repeat(colors, [len(pop) for pop in pop_store_results])

    # Now 'colors' contains a unique color (population index) for each individual across all populations

    # Plotting
    plt.scatter(makespans, final_energy, c=colors, cmap='viridis', marker='o', label='Pareto Front')
    plt.xlabel('Makespan')
    plt.ylabel('Energy')
    plt.title('Pareto Front across Populations')
    plt.pause(0.01)
    '''

total_time = round(current_time - old_time, 2)
print("Total time:", total_time, "s")

gantt(result_sch)

pareto_data = pareto_front(data_gen_store[-1])

df_pareto = pd.DataFrame(pareto_data, columns=['Makespan', 'Energy'])
df_iteration = pd.DataFrame(iteration, columns=['Time', 'Best Makespan Value'])
df_iteration_energy = pd.DataFrame(iteration_energy, columns=['Time', 'Best Energy Value'])

# Remove rows where 'Energy Value' is 0
df_iteration_energy = df_iteration_energy.loc[df_iteration_energy['Best Energy Value'] != 0]

with pd.ExcelWriter('Graph Excel/Iterations_VNS_NSGAII.xlsx', engine='xlsxwriter') as writer:
    df_pareto.to_excel(writer, sheet_name='Pareto Front', index=False)
    df_iteration.to_excel(writer, sheet_name='Iteration Makespan', index=False)
    df_iteration_energy.to_excel(writer, sheet_name='Iteration Energy', index=False)
