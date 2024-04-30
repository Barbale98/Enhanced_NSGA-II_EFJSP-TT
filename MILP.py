# Gurobi: FJSP with transportation time with both energy and makespan objectives

# Time limit MILP MODEL
TimeLimit = 2000


import gurobipy as gp
import numpy as np
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from Data_reader import TP,TRT,PP,m,n,N,M_ij,AUX_power,AGV_power,IDLE_power

# Create a new model
model = gp.Model("EFJSS")
# Set a time limit (e.g., 60 seconds)
model.setParam('TimeLimit', TimeLimit)
# Set parameters
model.setParam('Threads', 8)
'''
model.setParam('Method', 0)  # primal simplex method for LP relacation
model.setParam('Threads', 8)
model.setParam('MIPFocus', 2)  # focus on finding feasible solutions
'''
# Parameters
H = 1000  # A large positive number
# Decision Variables
# Sij identifies the start time of assignment j of job i
S = {}
# Cij identifies the completion time of assignment j of job i
C = {}
# Xijk is 1 if assignment j of job i is performed by machine k
X = {}
# Tijqk is 1 if assignment j of job i is performed by machine k
T = {}
# Yiji_starj_stark is 1 if assignment j_star of job i_star follows assignment j of job i on machine k
Y = {}
# Variable Creation
for i in range(1, n+1):
    for j in range(1, N[i]+1):
        S[i, j] = model.addVar(vtype=GRB.INTEGER, name=f"S_{i}_{j}")
        C[i, j] = model.addVar(vtype=GRB.INTEGER, name=f"C_{i}_{j}")
        for k in M_ij[i,j]:
            X[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"X_{i}_{j}_{k}")
            for i_star in range(1, n + 1):
                for j_star in range(1, N[i_star] + 1):
                    if k in M_ij[i_star,j_star]:
                        Y[i, j, i_star, j_star, k] = model.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{j}_{i_star}_{j_star}_{k}")
            if(j>1):
                for q in M_ij[i,j-1]:
                        T[i, j, q, k] = model.addVar(vtype=GRB.BINARY, name=f"T_{i}_{j}_{q}_{k}")

Tmakespan = model.addVar(vtype=GRB.INTEGER, name="Tmakespan")
Etot = model.addVar(vtype=GRB.CONTINUOUS, name="Etot")
Eprod = model.addVar(vtype=GRB.CONTINUOUS, name="Eprod")
Eidle = model.addVar(vtype=GRB.CONTINUOUS, name="Eidle")
Eagv = model.addVar(vtype=GRB.CONTINUOUS, name="Eagv")
Eaux = model.addVar(vtype=GRB.CONTINUOUS, name="Eaux")

# FJSP with transportation time constraints
# Constraint (1)
for i in range(1, n + 1):
    for j in range(1, N[i] + 1):
         model.addConstr(gp.quicksum(X[i, j, k] for k in M_ij[i,j]) == 1, f"Constraint_1_{k}")
# Constraint (2)
for i in range(1, n + 1):
    for j in range(1, N[i] + 1):
        model.addConstr(C[i, j] == S[i, j] + gp.quicksum(TP[i, j, k] * X[i, j, k] for k in M_ij[i,j]),
                        f"Constraint_2_{i}_{j}")
# Constraint (3)
for i in range(1, n + 1):
    for j in range(1, N[i] + 1):
        for k in M_ij[i,j]:
            for i_star in range(1, n + 1):
                for j_star in range(1, N[i_star] + 1):
                    if i != i_star or j != j_star:
                        if k in M_ij[i_star,j_star]:
                            model.addConstr(C[i, j] - C[i_star, j_star] +
                                            H * (1 - X[i, j, k]) + H * (1 - X[i_star, j_star, k]) +
                                            H * Y[i, j, i_star, j_star, k] >= TP[i, j, k],
                                            f"Constraint_3_{i}_{j}_{k}_{i_star}_{j_star}")
# Constraint (4)
for i in range(1, n + 1):
    for j in range(1, N[i] + 1):
        for k in M_ij[i,j]:
            for i_star in range(1, n + 1):
                for j_star in range(1, N[i_star] + 1):
                    if i != i_star or j != j_star:
                        if k in M_ij[i_star, j_star]:
                            model.addConstr(C[i_star, j_star] - C[i, j] +
                                            H * (1 - X[i, j, k]) + H * (1 - X[i_star, j_star, k]) +
                                            H * (1 - Y[i, j, i_star, j_star, k]) >= TP[i_star, j_star, k],
                                            f"Constraint_4_{i}_{j}_{k}_{i_star}_{j_star}")
# Constraint (5)
for i in range(1, n + 1):
    for j in range(2, N[i] + 1):
        for k in M_ij[i,j]:
            for q in M_ij[i,j-1]:
                    model.addConstr(T[i, j, q, k] <= X[i, j, k], f"Constraint_T_One_1_{i}_{j}_{q}_{k}")
                    model.addConstr(T[i, j, q, k] <= X[i, j - 1, q], f"Constraint_T_One_2_{i}_{j}_{q}_{k}")
                    model.addConstr(T[i, j, q, k] >= X[i, j, k] + X[i, j-1, q] - 1, f"Constraint_T_One_3_{i}_{j}_{q}_{k}")
# Constraint (6)
for i in range(1, n + 1):
    for j in range(2, N[i] + 1):
        for k in M_ij[i,j]:
            for q in M_ij[i,j-1]:
                    model.addConstr(H * (1 - T[i, j, q, k]) + S[i, j] - C[i, j - 1] >= TRT[q, k],
                        f"Constraint_6_{i}_{j}_{k}_{q}")
# Constraint (7)
for i in range(1, n + 1):
    j=1
    for k in M_ij[i,j]:
        model.addConstr(H*(1-X[i,j,k]) + S[i, j] >= TRT[0,k], f"Constraint_first_transport_{i}_{j}_{k}")
# Constraint (8)
for i in range(1, n + 1):
    j=N[i]
    for k in M_ij[i,j]:
        model.addConstr(H*(1-X[i,j,k]) + Tmakespan >= C[i, j] + TRT[k,0], f"Constraint_last_transport_{i}_{j}_{k}")


# Energy constraints
# Constraint (9)
model.addConstr(Eprod == gp.quicksum((C[i, j] - S[i, j])/60 * X[i, j, k] * PP[(i,j,k)] for i in range(1, n + 1) for j in range(1, N[i] + 1) for k in M_ij[i,j]),f"ProductionEnergy")
# Constraint (10)
model.addConstr(Eidle == gp.quicksum(IDLE_power[k] / 60 * (Tmakespan - gp.quicksum((C[i, j] - S[i, j]) * X[i, j, k+1] for i in range(1, n + 1) for j in range(1, N[i] + 1) if k+1 in M_ij[i, j])) for k in range(m)),name="IdleEnergy")
# Constraint (11) WITH LAST AND FIRST TRANSPORTATION
model.addConstr(Eagv == gp.quicksum(TRT[q, k] / 60 * AGV_power*T[i, j, q, k] for i in range(1, n + 1) for j in range(2, N[i] + 1) for k in M_ij[i,j] for q in M_ij[i,j-1]) + gp.quicksum(TRT[0, k] / 60 * AGV_power * X[i, 1, k] for i in range(1, n + 1) for k in M_ij[i,1]) + gp.quicksum(TRT[q,0]/60*AGV_power*X[i, N[i], q] for i in range(1, n + 1) for q in M_ij[i,N[i]]),f"AGVEnergy")
# Constraint (12)
model.addConstr(Eaux == Tmakespan/60 *AUX_power,f"AUXEnergy")
# Constraint (13)
model.addConstr(Etot == Eaux + Eprod + Eagv + Eidle, "EnergyConstraint")


# Objective definition and priority
# First objective
model.setObjectiveN(Tmakespan, index=0, priority=10, name='obj1')
# Second objective
model.setObjectiveN(Etot, index=1, priority=1, name='obj2')

# Optimize model
model.optimize()

# Print the optimal solution
if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:

    print("Tmakespan: ",Tmakespan.x)
    print("Etot: ",round(Etot.x,2))
    print("Eprod: ",round(Eprod.x,2),"Eaux: ",round(Eaux.x,2),"Eidle: ",round(Eidle.x,2),"Eagv: ",round(Eagv.x,2))

    # Extract the schedule
    schedule = {i: {j: [] for j in range(1, N[i] + 1)} for i in range(1, n + 1)}
    schedule_agv = {i: {j: [] for j in range(1, N[i] + 2)} for i in range(1, n + 1)}
    for i in range(1, n + 1):
        for j in range(1, N[i] + 1):
            for k in M_ij[i,j]:
                if X[i, j, k].x > 0.5:
                    # ( macchina, start time, completion time)
                    schedule[i][j].append([k, S[i, j].x, C[i, j].x])
                    if j>1:
                        for q in M_ij[i,j-1]:
                            if X[i, j - 1, q].x > 0.5 and TRT[q, k] != 0:
                                # macchina dopo, completion time precedente, transportation time, machine after
                                schedule_agv[i][j].append([q, C[i, j - 1].x, TRT[q, k], k])

    # Add information about the first transportation to schedule_agv
    for i in range(1, n + 1):
        j = 1
        for k in M_ij[i, j]:
            if X[i, j, k].x > 0.5 and TRT[0, k] != 0:
                schedule_agv[i][j].append([0, S[i, j].x-TRT[0, k], TRT[0, k], k])

    # Add information about the last transportation to schedule_agv
    for i in range(1, n + 1):
        j = N[i]
        for k in M_ij[i, j]:
            if X[i, j, k].x > 0.5 and TRT[k, 0] != 0:
                schedule_agv[i][j+1].append([k, C[i, j].x, TRT[k, 0], 0])

    # Plot the Gantt chart
    fig, ax = plt.subplots()

    # colors
    unique_job_ids = set(range(1, n + 1))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_job_ids)))

    product_colors = {}  # Dictionary to store product ID-color mapping

    for jobs, color in zip(unique_job_ids, colors):
        product_colors[jobs] = color
    # Draws the gantt chart
    for i in range(1, n + 1):
        for j in range(1, N[i] + 1):
            for task in schedule[i][j]:
                ax.barh(task[0], task[2] - task[1], left=task[1], height=0.6, align='center', color=product_colors[i],
                        edgecolor='black',linewidth=0.3)
                ax.text(task[1] + (task[2] - task[1]) / 2, task[0], str(i)+ "," + str(j), ha='center', va='center',
                        color='black', fontsize=8)
    for i in range(1, n + 1):
        for j in range(2, N[i]+1):
            for move in schedule_agv[i][j]:
                ax.barh(move[0] + 0.4, move[2], left=move[1], height=0.2, align='center', color='orange',
                        edgecolor='black')
                ax.text(move[1] + move[2] / 2, move[0] + 0.4, str(i) + str(move[0]) + str(move[3]), ha='center',
                        va='center', color='black', fontsize=6)
    for i in range(1, n + 1):
        j=1
        for move in schedule_agv[i][j]:
            ax.barh(move[3] - 0.4, move[2], left=move[1], height=0.2, align='center', color='orange',
                    edgecolor='black')
            ax.text(move[1] + move[2] / 2, move[3] - 0.4, str(i) + 'LU' + str(move[3]), ha='center',
                    va='center', color='black', fontsize=6)
    for i in range(1, n + 1):
        j=N[i]
        for move in schedule_agv[i][j+1]:
            ax.barh(move[0] - 0.4, move[2], left=move[1], height=0.2, align='center', color='orange',
                    edgecolor='black')
            ax.text(move[1] + move[2] / 2, move[0] - 0.4, str(i) + str(move[0]) + 'LU', ha='center',
                    va='center', color='black', fontsize=6)

    # Determine the locator parameters based on the makespan
    if Tmakespan.x <= 100:
        major_tick_locator = 5
        minor_tick_locator = 1
    elif Tmakespan.x <= 200:  # Adjust these ranges as needed
        major_tick_locator = 10
        minor_tick_locator = 5
    elif Tmakespan.x <= 400:  # Adjust these ranges as needed
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

    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    # Set x-axis limit to start at 0
    ax.set_xlim(0, Tmakespan.x+2)
    ax.set_yticks(range(1, m+1))
    ax.set_yticklabels([f"M{i}" for i in range(1, m+1)])
    ax.set_title(f'Gantt Chart')

    plt.show()

else:
    print('No solution found')
