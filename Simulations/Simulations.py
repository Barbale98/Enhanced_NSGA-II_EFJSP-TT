import pandas as pd

def simulation(file_name,file_name_Ext):
    from sim_NSGAII import run_simulation
    from sim_VNSNSGAII import run_simulation as run_simulation2
    from sim_EVNSNSGAII import run_simulation as run_simulation3

    def save(num_run,file_name,file_name_Ext):
        parameters_list = {}

        total_time_E, min_makespan_E, min_energy_E, iteration_energy, iteration, pareto_data, pareto_number_E = run_simulation3(file_name)

        pareto_data_list3 = pd.DataFrame(pareto_data, columns=['EVNS-NSGA-II', 'EVNS-NSGA-II'])
        iteration_data_list3 = pd.DataFrame(iteration, columns=['Time EVNS-NSGA-II', 'Makespan EVNS-NSGA-II'])
        iteration_energy_data_list3 = pd.DataFrame(iteration_energy, columns=['Time EVNS-NSGA-II', 'Energy EVNS-NSGA-II'])
        iteration_energy_data_list3 = iteration_energy_data_list3.loc[
            iteration_energy_data_list3['Energy EVNS-NSGA-II'] != 0]

        total_time, min_makespan, min_energy, iteration_energy, iteration, pareto_data, pareto_number, gen = run_simulation(total_time_E,file_name)
        parameters_list.update({
            'NSGA-II_makespan': min_makespan,
            'NSGA-II_energy': min_energy,
            'NSGA-II_time': total_time,
            'NSGA-II_pareto number': pareto_number,
            'NSGA-II_gen': gen
        })
        pareto_data_list1 = pd.DataFrame(pareto_data, columns=['NSGA-II', 'NSGA-II'])
        iteration_data_list1 = pd.DataFrame(iteration, columns=['Time NSGA-II', 'Makespan NSGA-II'])
        iteration_energy_data_list1 = pd.DataFrame(iteration_energy, columns=['Time NSGA-II', 'Energy NSGA-II'])
        iteration_energy_data_list1 = iteration_energy_data_list1.loc[iteration_energy_data_list1['Energy NSGA-II'] != 0]

        total_time, min_makespan, min_energy, iteration_energy, iteration, pareto_data, pareto_number, gen = run_simulation2(total_time_E, file_name)
        parameters_list.update({
            'VNS-NSGA-II_makespan': min_makespan,
            'VNS-NSGA-II_energy': min_energy,
            'VNS-NSGA-II_time': total_time,
            'VNS-NSGA-II_pareto number': pareto_number,
            'VNS-NSGA-II_gen': gen
        })
        pareto_data_list2 = pd.DataFrame(pareto_data, columns=['VNS-NSGA-II', 'VNS-NSGA-II'])
        iteration_data_list2 = pd.DataFrame(iteration, columns=['Time VNS-NSGA-II', 'Makespan VNS-NSGA-II'])
        iteration_energy_data_list2 = pd.DataFrame(iteration_energy, columns=['Time VNS-NSGA-II', 'Energy VNS-NSGA-II'])
        iteration_energy_data_list2 = iteration_energy_data_list2.loc[iteration_energy_data_list2['Energy VNS-NSGA-II'] != 0]

        parameters_list.update({
            'EVNS-NSGA-II_makespan': min_makespan_E,
            'EVNS-NSGA-II_energy': min_energy_E,
            'EVNS-NSGA-II_time': total_time_E,
            'EVNS-NSGA-II_pareto number': pareto_number_E
        })

        max_rows = max(pareto_data_list1.shape[0], pareto_data_list2.shape[0], pareto_data_list3.shape[0])
        standard_index = pd.Index(range(max_rows))
        pareto_df1 = pareto_data_list1.set_index(standard_index[:pareto_data_list1.shape[0]]).reindex(standard_index)
        pareto_df2 = pareto_data_list2.set_index(standard_index[:pareto_data_list2.shape[0]]).reindex(standard_index)
        pareto_df3 = pareto_data_list3.set_index(standard_index[:pareto_data_list3.shape[0]]).reindex(standard_index)
        combined_pareto_df = pd.concat([pareto_df1, pareto_df2, pareto_df3], axis=1)

        max_rows = max(iteration_data_list1.shape[0], iteration_data_list2.shape[0], iteration_data_list3.shape[0])
        standard_index = pd.Index(range(max_rows))
        iteration_df1 = iteration_data_list1.set_index(standard_index[:iteration_data_list1.shape[0]]).reindex(standard_index)
        iteration_df2 = iteration_data_list2.set_index(standard_index[:iteration_data_list2.shape[0]]).reindex(standard_index)
        iteration_df3 = iteration_data_list3.set_index(standard_index[:iteration_data_list3.shape[0]]).reindex(standard_index)
        combined_iteration_df = pd.concat([iteration_df1, iteration_df2, iteration_df3], axis=1)

        max_rows = max(iteration_energy_data_list1.shape[0], iteration_energy_data_list2.shape[0], iteration_energy_data_list3.shape[0])
        standard_index = pd.Index(range(max_rows))
        iteration_energy_df1 = iteration_energy_data_list1.set_index(standard_index[:iteration_energy_data_list1.shape[0]]).reindex(standard_index)
        iteration_energy_df2 = iteration_energy_data_list2.set_index(standard_index[:iteration_energy_data_list2.shape[0]]).reindex(standard_index)
        iteration_energy_df3 = iteration_energy_data_list3.set_index(standard_index[:iteration_energy_data_list3.shape[0]]).reindex(standard_index)
        combined_iteration_energy_df = pd.concat([iteration_energy_df1, iteration_energy_df2, iteration_energy_df3], axis=1)
        path = "Final_test/"+file_name_Ext+"_"+num_run+".xlsx"
        print(path)
        excel_writer = pd.ExcelWriter(path, engine='xlsxwriter')

        # Saving the dataframes to the Excel file under different sheets for the first simulation
        combined_pareto_df.to_excel(excel_writer, sheet_name='Pareto', index=False)
        combined_iteration_df.to_excel(excel_writer, sheet_name='Iterations', index=False)
        combined_iteration_energy_df.to_excel(excel_writer, sheet_name='Iteration Energy', index=False)
        pd.DataFrame([parameters_list]).to_excel(excel_writer, sheet_name='Parameters', index=False)

        # Now you need to create a chart. For that, you need to access the workbook and worksheet objects.
        workbook = excel_writer.book
        worksheet = excel_writer.sheets['Pareto']

        # Create a scatter chart object
        chart = workbook.add_chart({'type': 'scatter'})

        # Define the color for each data set
        colors = ['#F4C343', '#52ADE6', '#E83423']

        # Assuming 'Makespan NSGA' is in the first column, 'Energy NSGA' is in the second column, and so on.
        for i, color in zip(range(0, combined_pareto_df.shape[1], 2), colors):
            chart.add_series({
                'name': f'=Pareto!${chr(65 + i)}$1',
                'categories': f'=Pareto!${chr(65 + i)}$2:${chr(65 + i)}${combined_pareto_df.shape[0] + 1}',
                'values': f'=Pareto!${chr(66 + i)}$2:${chr(66 + i)}${combined_pareto_df.shape[0] + 1}',
                'marker': {
                    'type': 'circle',
                    'size': 5,
                    'border': {'color': color},
                    'fill': {'color': color}
                },
            })
        # Set legend position
        chart.set_legend({'position': 'top'})
        # Configure the chart axes.
        chart.set_x_axis({'name': 'Makespan (min)'})
        chart.set_y_axis({'name': 'Energy (kWh)'})
        chart.set_x_axis({
            'name': 'Makespan (min)',
            'major_gridlines': {
                'visible': True,
                'line': {'color': '#D9D9D9'}  # Light gray color for major gridlines
            }
        })

        chart.set_y_axis({
            'name': 'Energy (kWh)',
            'major_gridlines': {
                'visible': True,
                'line': {'color': '#D9D9D9'}
            }})
        # Insert the chart into the worksheet.
        worksheet.insert_chart('K1', chart)

        # Access the workbook and worksheet objects
        workbook = excel_writer.book
        worksheet = excel_writer.sheets['Iterations']

        # Create a scatter plot chart object
        chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})

        # Configure the series for the chart from the 'Iterations' data
        # Assuming your data starts in row 2 (Excel row 1), and the columns are A, B for NSGA-II, C, D for VNS-NSGA-II, E, F for EVNS-NSGA-II
        chart.add_series({
            'name': 'NSGA-II',
            'categories': '=Iterations!$A$2:$A$20000',
            'values': '=Iterations!$B$2:$B$20000',
            'marker': {'type': 'none'},
            'line': {'color': '#F4C343'},
        })

        chart.add_series({
            'name': 'VNS-NSGA-II',
            'categories': '=Iterations!$C$2:$C$20000',
            'values': '=Iterations!$D$2:$D$20000',
            'marker': {'type': 'none'},
            'line': {'color': '#52ADE6'},
        })

        chart.add_series({
            'name': 'EVNS-NSGA-II',
            'categories': '=Iterations!$E$2:$E$20000',
            'values': '=Iterations!$F$2:$F$20000',
            'marker': {'type': 'none'},
            'line': {'color': '#E83423'},
        })
        # Set legend position
        chart.set_legend({'position': 'top'})
        # Configure the chart axes
        chart.set_x_axis({'name': 'Time (s)'})
        chart.set_y_axis({'name': 'Makespan (min)'})
        chart.set_x_axis({
            'name': 'Time (s)',
            'major_gridlines': {
                'visible': True,
                'line': {'color': '#D9D9D9'}  # Light gray color for major gridlines
            }
        })

        chart.set_y_axis({
            'name': 'Makespan (min)',
            'major_gridlines': {
                'visible': True,
                'line': {'color': '#D9D9D9'}
            }})
        # Insert the chart into the worksheet
        worksheet.insert_chart('H2', chart)

        # Access the workbook and worksheet objects
        workbook = excel_writer.book
        worksheet = excel_writer.sheets['Iteration Energy']

        # Create a scatter plot chart object
        chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})

        # Configure the series for the chart from the 'Iterations' data
        # Assuming your data starts in row 2 (Excel row 1), and the columns are A, B for NSGA-II, C, D for VNS-NSGA-II, E, F for EVNS-NSGA-II
        chart.add_series({
            'name': 'NSGA-II',
            'categories': "='Iteration Energy'!$A$2:$A$20000",
            'values': "='Iteration Energy'!$B$2:$B$20000",
            'marker': {'type': 'none'},
            'line': {'color': '#F4C343'},
        })

        chart.add_series({
            'name': 'VNS-NSGA-II',
            'categories': "='Iteration Energy'!$C$2:$C$20000",
            'values': "='Iteration Energy'!$D$2:$D$20000",
            'line': {'color': '#52ADE6'},
            'marker': {'type': 'none'},
        })

        chart.add_series({
            'name': 'EVNS-NSGA-II',
            'categories': "='Iteration Energy'!$E$2:$E$20000",
            'values': "='Iteration Energy'!$F$2:$F$20000",
            'marker': {'type': 'none'},
            'line': {'color': '#E83423'},
        })
        # Set legend position
        chart.set_legend({'position': 'top'})
        # Configure the chart axes
        chart.set_x_axis({'name': 'Time (s)'})
        chart.set_y_axis({'name': 'Energy (kWh)'})

        chart.set_x_axis({
            'name': 'Time (s)',
            'major_gridlines': {
                'visible': True,
                'line': {'color': '#D9D9D9'}  # Light gray color for major gridlines
            }
        })

        chart.set_y_axis({
            'name': 'Energy (kWh)',
            'major_gridlines': {
                'visible': True,
                'line': {'color': '#D9D9D9'}
            }})

        # Insert the chart into the worksheet
        worksheet.insert_chart('M1', chart)

        excel_writer._save()

        return parameters_list

    parameters_list_1 = save("1", file_name,file_name_Ext)
    parameters_list_2 = save("2", file_name,file_name_Ext)
    parameters_list_3 = save("3", file_name,file_name_Ext)
    parameters_list_4 = save("4", file_name,file_name_Ext)
    parameters_list_5 = save("5", file_name,file_name_Ext)

    # Convert each dictionary to a DataFrame
    df1 = pd.DataFrame([parameters_list_1])
    df2 = pd.DataFrame([parameters_list_2])
    df3 = pd.DataFrame([parameters_list_3])
    df4 = pd.DataFrame([parameters_list_4])
    df5 = pd.DataFrame([parameters_list_5])
    
    # Concatenate all DataFrames into a single DataFrame
    summary_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

    # Calculate mean, min, and max for numeric columns and exclude 'Simulation' from these calculations
    mean_values = summary_df.describe().loc['mean']
    min_values = summary_df.describe().loc['min']
    max_values = summary_df.describe().loc['max']

    # Append these rows to the DataFrame
    summary_df = summary_df._append({**{'Simulation': 'Average'}, **mean_values}, ignore_index=True)
    summary_df = summary_df._append({**{'Simulation': 'Minimum'}, **min_values}, ignore_index=True)
    summary_df = summary_df._append({**{'Simulation': 'Maximum'}, **max_values}, ignore_index=True)

    # Save the concatenated DataFrame to an Excel file
    excel_writer = pd.ExcelWriter("Final_test/"+file_name_Ext+"_" +'Summary.xlsx', engine='xlsxwriter')
    summary_df.to_excel(excel_writer, sheet_name='Summary', index=False)
    excel_writer._save()

simulation("MyInstance/Thesis instances/Test.xlsx", "TC")


'''
simulation("MyInstance/Thesis instances/Data_medium4.xlsx", "T9_M4")
simulation("MyInstance/Thesis instances/Data_medium5.xlsx", "T9_M5")
'''

