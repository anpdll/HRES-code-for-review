## Loading necessary libraries
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import pickle
import glob
import numpy as np
from calendar import month_abbr
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.stats import linregress
from scipy.stats import entropy
from scipy.stats import gaussian_kde
from scipy.stats import pearsonr
from kneed import KneeLocator
import matplotlib.pyplot as plt

####################################
####################################

## Making Yearly Profile
voltage = 480
power_factor = 0.9

def process_csv_files(csv_pattern):
    """
    Reads multiple CSV files, removes the first two rows, adds headers, 
    and returns a list of processed pandas DataFrames.

    Args:
        csv_pattern (str): A glob pattern specifying the CSV files to process (e.g., '*.csv', 'data/csv_files/*.csv').

    Returns:
        list: A list of pandas DataFrames, one for each processed CSV file.  
              Returns an empty list if no files match the pattern.
    """

    csv_files = glob.glob(csv_pattern)
    dataframes = []

    for file in csv_files:
        try:
            # Read the CSV, skipping the first two rows, and setting the header
            df = pd.read_csv(file, skiprows=2, header=None, names=['date', 'current', 'start', 'stop'])
            
            # Calculate the 'power' column
            df['power'] = np.sqrt(3) * voltage * power_factor * df['current'] / 1000

            # Remove the 'current', 'start' and 'stop' columns
            df = df.drop(columns=['current', 'start', 'stop'])

            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y %I:%M:%S %p')

            reindexed_df = trim_rearrange_and_reindex(df)
                       
            dataframes.append(reindexed_df)

        except pd.errors.EmptyDataError:  # Handle empty files or files with only skipped rows
            print(f"Warning: File '{file}' is empty or has only header rows after skipping. Skipping this file.")
        except Exception as e:  # Handle other potential errors during file reading 
            print(f"Error reading file or processing file '{file}': {e}")


    return dataframes

def trim_rearrange_and_reindex(df):
    """
    Trims DataFrame to exactly two weeks, rearranges to start from Monday,
    and reindexes dates to start from the first day of the month.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with 'date' column in datetime format
    
    Returns:
    pandas.DataFrame: Rearranged and reindexed DataFrame
    """
    # Sort DataFrame by date to ensure chronological order
    df = df.sort_values('date')
    
    # Find the first midnight after the start date
    start_date = df['date'].iloc[0].normalize() + pd.Timedelta(days=1)
    
    # Calculate the end date (start_date + 13 days, 23 hours, 59 minutes)
    end_date = start_date + pd.Timedelta(days=13, hours=23, minutes=59)
    
    # Filter the DataFrame to two weeks
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    trimmed_df = df.loc[mask].copy()
    
    # Get the day of week for the start date (0=Monday, 6=Sunday)
    start_day_of_week = start_date.weekday()
    
    if start_day_of_week != 0:  # If not starting on Monday
        # Split the DataFrame into two parts
        cutoff_date = end_date - pd.Timedelta(days=start_day_of_week)
        
        # Split the dataframe
        first_part = trimmed_df[trimmed_df['date'] <= cutoff_date]
        last_part = trimmed_df[trimmed_df['date'] > cutoff_date]
        
        # Concatenate the parts in the desired order
        rearranged_df = pd.concat([last_part, first_part])
        rearranged_df = rearranged_df.reset_index(drop=True)
    else:
        rearranged_df = trimmed_df
    
    # Get the time delta between consecutive timestamps to maintain the same frequency
    time_delta = rearranged_df['date'].diff().mode()[0]
    
    # Create new date range starting from first of the january
    new_dates = pd.date_range(
        start='2020-01-01',
        periods=len(rearranged_df),
        freq=time_delta
    )
   
    # first_of_month = rearranged_df['date'].iloc[0].replace(day=1)
    # new_dates = pd.date_range(
    #     start=first_of_month,
    #     periods=len(rearranged_df),
    #     freq=time_delta
    # )
    
    # Create new DataFrame with reindexed dates
    reindexed_df = rearranged_df.copy()
    #reindexed_df['original_date'] = reindexed_df['date']  # Keep original dates if needed
    reindexed_df['date'] = new_dates
   
    return reindexed_df


# Usage (assuming CSV files are in the current directory):
dataframes_list = process_csv_files('logged_csv/*.csv')  # Or provide a more specific path


def convert_to_hourly_resolution(dataframes_list):
    """
    Converts each DataFrame in the list to hourly resolution, averaging the 'power' values for each hour.

    Args:
        dataframes_list (list): A list of pandas DataFrames.

    Returns:
        list: A list of pandas DataFrames, one for each processed DataFrame.
    """
    hourly_dataframes = []

    for df in dataframes_list:
        # Resample the DataFrame to hourly resolution, averageing the 'power' values
        hourly_df = df.set_index('date').resample('h').mean().reset_index()
        hourly_dataframes.append(hourly_df)

    return hourly_dataframes

# Print hourly resolution dataframes
hourly_dataframes_list = convert_to_hourly_resolution(dataframes_list)

# load the monthly power consumption data
monthly_load = pd.read_excel('1051.xlsx') # has column 'month' (as 1, 2...) and 'total_kwh'

# print a bar chart of the monthly power consumption
plt.figure(figsize=(12, 6))
plt.bar(monthly_load['month'], monthly_load['total_kwh'], color='skyblue')
plt.xlabel('Month')
plt.ylabel('Total Power Consumption (kWh)')
plt.title('Monthly Power Consumption')
plt.xticks(monthly_load['month'], month_abbr[1:])
plt.show()


# Create day points including day 0 and day 365
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
mid_month_days = np.cumsum([0] + days_in_month[:-1]) + np.array(days_in_month) / 2

# Add points at the very beginning and end of the year
interpolation_days = np.array([0] + list(mid_month_days) + [365])

# Create periodic data by repeating appropriate values
monthly_values = monthly_load['total_kwh'].values
# Use average of first and last month for the endpoints
start_end_value = (monthly_values[0] + monthly_values[-1]) / 2
interpolation_values = np.array([start_end_value] + list(monthly_values) + [start_end_value])

# Create cubic spline interpolation
cs = CubicSpline(interpolation_days, interpolation_values, bc_type='periodic')

# Create daily points
daily_points = np.arange(1, 366)
daily_load = cs(daily_points)

# Create new DataFrame with daily values
daily_load_df = pd.DataFrame({
    'day': daily_points,
    'total_kwh': daily_load
})


# Calculate the average daily power consumption
average_daily_power = np.mean(daily_load)

# Create a mask for weekends (assuming day 1 is Monday)
# Saturday is day 6, 13, 20, etc. (day % 7 == 6)
# Sunday is day 7, 14, 21, etc. (day % 7 == 0)
weekend_mask = np.logical_or(daily_points % 7 == 6, daily_points % 7 == 0)

# Set weekend days to 0.2 of average daily power
daily_load[weekend_mask] = 0.2 * average_daily_power

daily_total =  daily_load.sum()
monthly_total = monthly_load['total_kwh'].sum()
scale_d_m = monthly_total/daily_total

daily_load = daily_load * scale_d_m
daily_load_df = daily_load_df['total_kwh'] * scale_d_m


# Create hourly timestamps for a year
year_start = pd.Timestamp('2024-01-01')  # Starting from a Monday
hourly_timestamps = pd.date_range(start=year_start, periods=365*24, freq='h')
yearly_profile = pd.DataFrame(index=hourly_timestamps, columns=['power'])
yearly_profile_no_noise = yearly_profile.copy()


def extract_typical_pattern(hourly_dataframes_list):
    # Combine all equipment patterns
    combined_pattern = pd.DataFrame(index=hourly_dataframes_list[0].index)
    
    for df in hourly_dataframes_list:
        equipment_name = f'equipment_{len(combined_pattern.columns)}'
        combined_pattern[equipment_name] = df['power']
    
    # Sum up all equipment
    combined_pattern['total_logged'] = combined_pattern.sum(axis=1)
    return combined_pattern

def generate_yearly_profile(daily_load, typical_pattern, yearly_timestamps):
    yearly_profile = pd.DataFrame(index=yearly_timestamps, columns=['power'])
    
    # Get the typical weekday and weekend patterns
    two_week_pattern = typical_pattern['total_logged'].values.reshape(14, 24)
    weekday_pattern = np.mean(two_week_pattern[:5], axis=0)  # Monday-Friday
    weekend_pattern = np.mean(two_week_pattern[5:7], axis=0)  # Saturday-Sunday
    
    # Calculate typical daily total for logged equipment
    typical_weekday_total = np.sum(weekday_pattern)
    
    for day in range(365):
        day_total = daily_load[day]
        day_start_hour = day * 24
        day_end_hour = (day + 1) * 24
        
        # Determine if it's a weekend
        is_weekend = (day + 1) % 7 in [0, 6]
        
        if is_weekend:
            # Weekend profile: 20% of average during all hours
            base_pattern = np.ones(24) * (day_total / 24)
            # Add small fluctuations even during weekends
            base_pattern += weekend_pattern * 0.1  # Small influence from logged pattern
        else:
            # Weekday profile
            production_hours = slice(7, 18)
            non_production_hours = list(range(0, 7)) + list(range(18, 24))
            
            total_production = 0.75 * day_total
            total_non_production = 0.25 * day_total
            
            # Create base load pattern
            base_pattern = np.zeros(24)
            base_pattern[production_hours] = total_production / 11
            base_pattern[non_production_hours] = total_non_production / 13
            
            # Increase the influence of logged pattern
            pattern_scale = day_total / typical_weekday_total
            # Increase from 0.3 to 0.7 for more fluctuation
            base_pattern += weekday_pattern * pattern_scale * 0.7
            base_pattern_no_noise = base_pattern.copy()

            # Add random fluctuations to make it more realistic
            np.random.seed(day)  
            noise = np.random.normal(0, base_pattern.mean() * 0.05, 24)  # 5% random variation
            base_pattern += noise
           
            # Ensure we maintain the daily total
            base_pattern = base_pattern * (day_total / base_pattern.sum())
            base_pattern_no_noise = base_pattern_no_noise * (day_total / base_pattern_no_noise.sum())
            
            # Ensure non-production hours don't go below minimum threshold
            min_power = 0.15 * base_pattern.mean()
            base_pattern[base_pattern < min_power] = min_power

            min_power_no_noise = 0.15 * base_pattern_no_noise.mean()
            base_pattern_no_noise[base_pattern_no_noise < min_power_no_noise] = min_power_no_noise
        
        
        
        yearly_profile.iloc[day_start_hour:day_end_hour, 0] = base_pattern
        yearly_profile_no_noise.iloc[day_start_hour:day_end_hour, 0] = base_pattern_no_noise
    
    return yearly_profile, yearly_profile_no_noise


# Extract typical pattern
typical_pattern = extract_typical_pattern(hourly_dataframes_list)

# Generate yearly profile
yearly_power_profile, yearly_power_profile_no_noise = generate_yearly_profile(daily_load, typical_pattern, hourly_timestamps)


####################################
####################################

## Running Optimization Model
# Initialize the model
model = gp.Model("Energy Optimization")

# Setting up the parameters
# PV
f_pv, alpha_pv, NOCT, G_std, T_cell_std = 0.86, -0.00386, 46, 1, 25

# Electrolyser
nel, Pel_min, Pel_max = 0.769, 0.1, 1

# Hydrogen Tank
nht, Eloht_min, Eloht_max, Eloht_in = 0.95, 0.01, 0.95, 0.5

# Fuel Cell
nfc, Pfc_min, Pfc_max= 0.75, 0.1, 1

n_inv = 0.95

# Grid
emission_factor, grid_per_kwh = 0.453592 * 1.0461, 0.068076

# CAPEX and OPEX
pv_capex, pv_opex_rate = 1181, 0.01  # combined with inverter cost as they both have same opex rate it is okay to combine them together
el_capex, el_opex_rate = 705, 0.075  
el_stack_replace_rate = 0.35  # 35% of CAPEX
ht_capex, ht_opex_rate = 480, 0.02  
fc_capex, fc_opex_rate = 2254, 0.03
fc_stack_replace_rate = 0.267 
inverter_capex = 0 
replacement_year = 10 


# Economic/Misc
N, r, bigM = 20, 0.1, 1000000
discount_factors = [(1 + r)**(n) for n in range(1, N+1)]


# Importing weather data
# Weather file
data_pv = pd.read_csv('weather_year.csv')
GHI = data_pv['GHI']
Tamb = data_pv['Temperature']
T_cell = Tamb + (GHI / 800) * (NOCT - 20)

nPeriods = len(GHI)
nPeriodsplus1 = nPeriods + 1


# Load file
# create a empty dataframe P_load with column as load
P_load = pd.DataFrame(columns=['Load'])
# Load yearly profile to P_load
P_load['Load'] = yearly_power_profile['power']
P_load = P_load['Load']
P_load = P_load/1000 # creating power in mega watts
# convert P_load datatype to float
P_load = P_load.astype(float)
# Max P_load value
P_load_max = P_load.max()


# Defining rated capacity bounds
# Variable Bounds
Ppv_rated_min, Ppv_rated_max = 0, 200
Pel_rated_min, Pel_rated_max = 0, 100
Eloht_rated_min, Eloht_rated_max = 0, 200
Pfc_rated_min, Pfc_rated_max = 0, 100

# Setting Solver Parameters and Starting Optimization
def create_payoff_table(original_model):
    """Creates payoff table by optimizing each objective individually"""
    
    # First optimize for cost
    cost_model = gp.Model("Cost Model")
    
    # Create variables in the new model
    vars_dict = create_variables(cost_model)
    
    # Add all constraints
    add_all_constraints(cost_model, vars_dict)
    
    # Create cost objective
    total_cost = get_total_cost_expr(vars_dict)
    
    # Minimize cost
    cost_model.setObjective(total_cost, GRB.MINIMIZE)
    cost_model.optimize()
    
    if cost_model.status == GRB.OPTIMAL:
        cost_min = cost_model.objVal
        emission_at_cost_min = get_emission_value(cost_model, vars_dict)
    else:
        raise Exception("Could not find optimal solution for cost minimization")
    
    # Then optimize for emissions
    emission_model = gp.Model("Emission Model")
    
    # Create variables in the new model
    vars_dict = create_variables(emission_model)
    
    # Add all constraints
    add_all_constraints(emission_model, vars_dict)
    
    annual_emission_expr = get_emission_expr(vars_dict)
    
    emission_model.setObjective(annual_emission_expr, GRB.MINIMIZE)
    emission_model.optimize()
    
    if emission_model.status == GRB.OPTIMAL:
        emission_min = emission_model.objVal
        cost_at_emission_min = get_total_cost_value(emission_model, vars_dict)
    else:
        raise Exception("Could not find optimal solution for emission minimization")
    
    return {
        'cost': {'min': cost_min, 'max': cost_at_emission_min},
        'emission': {'min': emission_min, 'max': emission_at_cost_min}
    }

def create_variables(model):
    """Create all variables for a new model"""
    # Create all variables as in your original model
    del1 = model.addVars(nPeriods, vtype=GRB.BINARY, name="del1")
    del2 = model.addVars(nPeriods, vtype=GRB.BINARY, name="del2")
    del3 = model.addVars(nPeriods, vtype=GRB.BINARY, name="del3")
    
    P_pv = model.addVars(nPeriods, lb=0, name="P_pv")
    P_el = model.addVars(nPeriods, lb=0, name="P_el")
    Eloht = model.addVars(nPeriodsplus1, lb=0, name="Eloht")
    P_fc = model.addVars(nPeriods, lb=0, name="P_fc")
    P_grid = model.addVars(nPeriods, lb=0, name="P_grid")
    
    Ppv_rated = model.addVar(lb=Ppv_rated_min, ub=Ppv_rated_max, name="Ppv_rated")
    Pel_rated = model.addVar(lb=Pel_rated_min, ub=Pel_rated_max, name="Pel_rated")
    Eloht_rated = model.addVar(lb=Eloht_rated_min, ub=Eloht_rated_max, name="Eloht_rated")
    Pfc_rated = model.addVar(lb=Pfc_rated_min, ub=Pfc_rated_max, name="Pfc_rated")
    
    Pel_rated_aux = model.addVars(nPeriods, lb=0, name="Pel_rated_aux")
    Pfc_rated_aux = model.addVars(nPeriods, lb=0, name="Pfc_rated_aux")
    Pgrid_rated_aux = model.addVars(nPeriods, lb=0, name="Pgrid_rated_aux")
    
    model.update()
    
    # Return dictionary of all variables
    return {v.VarName: v for v in model.getVars()}


def augmecon2_optimization(original_model, payoff_table, grid_points=5, r=1e-3):
    emission_range = payoff_table['emission']['max'] - payoff_table['emission']['min']
    step = emission_range / (grid_points - 1)
    
    results = []
    detailed_results = []
    
    for i in range(grid_points):
        print(f"Solving iteration {i+1}/{grid_points}")
        
        current_model = gp.Model(f"AUGMECON2_iteration_{i}")
        
        # Create variables
        vars_dict = create_variables(current_model)
        
        # Add all constraints
        add_all_constraints(current_model, vars_dict)
        
        # Calculate emission limit for this iteration
        emission_limit = payoff_table['emission']['max'] - i * step
        surplus = current_model.addVar(name='surplus')
        
        # Add emission constraint
        annual_emission_expr = get_emission_expr(vars_dict)
        current_model.addConstr(annual_emission_expr + surplus == emission_limit,
                              name='emission_constraint')
        
        # Set objective
        total_cost_expr = get_total_cost_expr(vars_dict)
        current_model.setObjective(total_cost_expr + r * surplus, GRB.MINIMIZE)
        
        # Set solver parameters
        current_model.setParam('MIPGap', 0.01)  # 1% optimality gap
        current_model.setParam('TimeLimit', 3600)  # 1 hr time limit
        
        current_model.optimize()
        
        if current_model.status == GRB.OPTIMAL:
            # Store basic results
            basic_result = {
                'emission_limit': emission_limit,
                'actual_emission': get_emission_value(current_model, vars_dict),
                'total_cost': get_total_cost_value(current_model, vars_dict),
                'surplus': surplus.x,
                'Ppv_rated': vars_dict['Ppv_rated'].x,
                'Pel_rated': vars_dict['Pel_rated'].x,
                'Eloht_rated': vars_dict['Eloht_rated'].x,
                'Pfc_rated': vars_dict['Pfc_rated'].x
            }
            results.append(basic_result)

            # Store detailed results
            detailed_result = {
                'iteration': i,
                'basic_results': basic_result,
                'time_series': {
                    'P_pv': [vars_dict[f'P_pv[{t}]'].x for t in range(nPeriods)],
                    'P_el': [vars_dict[f'P_el[{t}]'].x for t in range(nPeriods)],
                    'P_fc': [vars_dict[f'P_fc[{t}]'].x for t in range(nPeriods)],
                    'P_grid': [vars_dict[f'P_grid[{t}]'].x for t in range(nPeriods)],
                    'Eloht': [vars_dict[f'Eloht[{t}]'].x for t in range(nPeriodsplus1)]
                },
                'binary_vars': {
                    'del1': [vars_dict[f'del1[{t}]'].x for t in range(nPeriods)],
                    'del2': [vars_dict[f'del2[{t}]'].x for t in range(nPeriods)],
                    'del3': [vars_dict[f'del3[{t}]'].x for t in range(nPeriods)]
                }
            }
            detailed_results.append(detailed_result)

            print(f"Found solution with cost: {results[-1]['total_cost']:.2f}, " 
                  f"emission: {results[-1]['actual_emission']:.2f}")
        else:
            print(f"Could not find optimal solution for iteration {i+1}")

    # Save all results
    results_df = pd.DataFrame(results)

    # Sort results by emission
    results_df = results_df.sort_values('actual_emission')
    
    # Remove dominated solutions
    non_dominated = []
    min_cost = float('inf')
    
    for idx, row in results_df.iterrows():
        if row['total_cost'] < min_cost:
            non_dominated.append(row)
            min_cost = row['total_cost']
    
    final_results = pd.DataFrame(non_dominated)
    
    # Save to files
    final_results.to_csv('optimization_results.csv')
    with open('detailed_optimization_results.pkl', 'wb') as f:
        pickle.dump({'basic_results': final_results,
                    'detailed_results': detailed_results,
                    'payoff_table': payoff_table}, f)
    
    return final_results


def add_all_constraints(model, vars_dict):
    """Add all constraints to the model"""
    
    # Power Balance Constraint
    model.addConstrs(( n_inv * vars_dict[f'P_pv[{t}]'] + vars_dict[f'P_grid[{t}]'] + n_inv * vars_dict[f'P_fc[{t}]'] >= 
                      vars_dict[f'P_el[{t}]'] + P_load[t] for t in range(nPeriods)), "constraint1")

    # PV constraint
    model.addConstrs((vars_dict[f'P_pv[{t}]'] == f_pv * (vars_dict['Ppv_rated'] * 1e6) * 
                      (GHI[t] / (G_std * 1000)) * (1 + alpha_pv * (T_cell[t] - T_cell_std)) / 1e6 
                      for t in range(nPeriods)), "constraint2")

    # Electrolyser Constraints
    model.addConstrs((vars_dict[f'Pel_rated_aux[{t}]'] <= vars_dict['Pel_rated'] - 
                      (1 - vars_dict[f'del1[{t}]']) * Pel_rated_min 
                      for t in range(nPeriods)), "constraint3")
    
    model.addConstrs((vars_dict[f'Pel_rated_aux[{t}]'] >= vars_dict['Pel_rated'] - 
                      (1 - vars_dict[f'del1[{t}]']) * Pel_rated_max 
                      for t in range(nPeriods)), "constraint4")
    
    model.addConstrs((vars_dict[f'Pel_rated_aux[{t}]'] <= Pel_rated_max * 
                      vars_dict[f'del1[{t}]'] for t in range(nPeriods)), "constraint5")
    
    model.addConstrs((vars_dict[f'Pel_rated_aux[{t}]'] >= Pel_rated_min * 
                      vars_dict[f'del1[{t}]'] for t in range(nPeriods)), "constraint6")

    model.addConstrs((vars_dict[f'P_el[{t}]'] >= Pel_min * 
                      vars_dict[f'Pel_rated_aux[{t}]'] for t in range(nPeriods)), "constraint7")
    
    model.addConstrs((vars_dict[f'P_el[{t}]'] <= Pel_max * 
                      vars_dict[f'Pel_rated_aux[{t}]'] for t in range(nPeriods)), "constraint8")

    # Fuel Cell Constraint
    model.addConstrs((vars_dict[f'Pfc_rated_aux[{t}]'] <= vars_dict['Pfc_rated'] - 
                      (1 - vars_dict[f'del2[{t}]']) * Pfc_rated_min 
                      for t in range(nPeriods)), "constraint13")
    
    model.addConstrs((vars_dict[f'Pfc_rated_aux[{t}]'] >= vars_dict['Pfc_rated'] - 
                      (1 - vars_dict[f'del2[{t}]']) * Pfc_rated_max 
                      for t in range(nPeriods)), "constraint14")
    
    model.addConstrs((vars_dict[f'Pfc_rated_aux[{t}]'] <= Pfc_rated_max * 
                      vars_dict[f'del2[{t}]'] for t in range(nPeriods)), "constraint15")
    
    model.addConstrs((vars_dict[f'Pfc_rated_aux[{t}]'] >= Pfc_rated_min * 
                      vars_dict[f'del2[{t}]'] for t in range(nPeriods)), "constraint16")

    model.addConstrs((vars_dict[f'P_fc[{t}]'] >= Pfc_min * 
                      vars_dict[f'Pfc_rated_aux[{t}]'] for t in range(nPeriods)), "constraint17")
    
    model.addConstrs((vars_dict[f'P_fc[{t}]'] <= Pfc_max * 
                      vars_dict[f'Pfc_rated_aux[{t}]'] for t in range(nPeriods)), "constraint18")

    # Hydrogen Tank Constraint
    model.addConstr(vars_dict['Eloht[0]'] == vars_dict['Eloht_rated'] * Eloht_in, "constraint11")
    
    model.addConstrs((vars_dict[f'Eloht[{t + 1}]'] == vars_dict[f'Eloht[{t}]'] + 
                      vars_dict[f'P_el[{t}]'] * nel - 
                      (vars_dict[f'P_fc[{t}]']/(nfc*nht)) for t in range(nPeriods)), "constraint12")

    model.addConstrs((vars_dict[f'Eloht[{t}]'] >= Eloht_min * 
                      vars_dict['Eloht_rated'] for t in range(nPeriodsplus1)), "constraint9")
    
    model.addConstrs((vars_dict[f'Eloht[{t}]'] <= Eloht_max * 
                      vars_dict['Eloht_rated'] for t in range(nPeriodsplus1)), "constraint10")

    # Grid constraint
    model.addConstrs((vars_dict[f'Pgrid_rated_aux[{t}]'] == vars_dict[f'del3[{t}]'] * 
                      P_load_max for t in range(nPeriods)), "constraint19")

    model.addConstrs((vars_dict[f'P_grid[{t}]'] >= 0 for t in range(nPeriods)), "constraint20")
    
    model.addConstrs((vars_dict[f'P_grid[{t}]'] <= vars_dict[f'Pgrid_rated_aux[{t}]'] 
                      for t in range(nPeriods)), "constraint21")

    model.addConstrs(((n_inv * vars_dict[f'P_pv[{t}]'] + n_inv * vars_dict[f'P_fc[{t}]']) - P_load[t] <= 
                  bigM * (1 - vars_dict[f'del3[{t}]']) for t in range(nPeriods)), "constraint22")

    model.addConstrs((P_load[t] - (n_inv * vars_dict[f'P_pv[{t}]'] + n_inv * vars_dict[f'P_fc[{t}]']) <= 
                  bigM * vars_dict[f'del3[{t}]'] for t in range(nPeriods)), "constraint23")

    # Additional Constraints
    model.addConstrs((vars_dict[f'del1[{t}]'] + vars_dict[f'del2[{t}]'] <= 1 
                      for t in range(nPeriods)), "constraint24")
    
    model.addConstrs((vars_dict[f'del1[{t}]'] + vars_dict[f'del3[{t}]'] <= 1 
                      for t in range(nPeriods)), "constraint25")

    model.update()



def get_emission_value(model, vars_dict):
    """Calculate emission value using variable dictionary"""
    return sum(emission_factor * vars_dict[f'P_grid[{t}]'].x * 1000 
              for t in range(nPeriods))


def get_total_cost_value(model, vars_dict):
    # Base CAPEX
    total_capex = (
        pv_capex * vars_dict['Ppv_rated'].x * 1000 +
        el_capex * vars_dict['Pel_rated'].x * 1000 +
        ht_capex * vars_dict['Eloht_rated'].x * 1000 +
        fc_capex * vars_dict['Pfc_rated'].x * 1000 +
        inverter_capex * vars_dict['Ppv_rated'].x * 1000
    )

    # Annual OPEX
    total_opex = sum(
        (pv_opex_rate * pv_capex * vars_dict['Ppv_rated'].x * 1000 +
         el_opex_rate * el_capex * vars_dict['Pel_rated'].x * 1000 +
         ht_opex_rate * ht_capex * vars_dict['Eloht_rated'].x * 1000 +
         fc_opex_rate * fc_capex * vars_dict['Pfc_rated'].x * 1000
         ) / discount_factors[n]
        for n in range(N))

    # Replacement costs at year 10
    replacement_cost = (
        el_stack_replace_rate * el_capex * vars_dict['Pel_rated'].x * 1000 +
        fc_stack_replace_rate * fc_capex * vars_dict['Pfc_rated'].x * 1000
    ) / discount_factors[replacement_year-1]

    # Grid power costs
    Grid_power_cost = sum(grid_per_kwh * vars_dict[f'P_grid[{t}]'].x * 1000 
                         for t in range(nPeriods))
    Grid_power_cost_for_life = sum(Grid_power_cost / discount_factors[n]
                                  for n in range(N))

    return total_capex + total_opex + replacement_cost + Grid_power_cost_for_life



def get_emission_expr(vars_dict):
    """Create emission expression using model variables"""
    return gp.quicksum(emission_factor * vars_dict[f'P_grid[{t}]']*1000 
                      for t in range(nPeriods))


def get_total_cost_expr(vars_dict):
    # Base CAPEX
    total_capex = (
        pv_capex * vars_dict['Ppv_rated'] * 1000 +  # PV CAPEX
        el_capex * vars_dict['Pel_rated'] * 1000 +  # Electrolyser CAPEX
        ht_capex * vars_dict['Eloht_rated'] * 1000 +  # H2 Storage CAPEX
        fc_capex * vars_dict['Pfc_rated'] * 1000 +  # Fuel Cell CAPEX
        inverter_capex * vars_dict['Ppv_rated'] * 1000  # Inverter CAPEX
    )

    # Annual OPEX
    total_opex = gp.quicksum(
        (pv_opex_rate * pv_capex * vars_dict['Ppv_rated'] * 1000 +  # PV OPEX
         el_opex_rate * el_capex * vars_dict['Pel_rated'] * 1000 +  # Electrolyser OPEX
         ht_opex_rate * ht_capex * vars_dict['Eloht_rated'] * 1000 +  # H2 Storage OPEX
         fc_opex_rate * fc_capex * vars_dict['Pfc_rated'] * 1000  # Fuel Cell OPEX
         ) / discount_factors[n]
        for n in range(N))

    # Replacement costs at year 10
    replacement_cost = (
        el_stack_replace_rate * el_capex * vars_dict['Pel_rated'] * 1000 +  # Electrolyser stack
        fc_stack_replace_rate * fc_capex * vars_dict['Pfc_rated'] * 1000  # Fuel cell stack
    ) / discount_factors[replacement_year-1]  # Apply discount factor for year 10

    # Grid power costs
    Grid_power_cost = gp.quicksum(grid_per_kwh * vars_dict[f'P_grid[{t}]']*1000 
                                 for t in range(nPeriods))
    Grid_power_cost_for_life = gp.quicksum(Grid_power_cost / discount_factors[n]
                                          for n in range(N))

    return total_capex + total_opex + replacement_cost + Grid_power_cost_for_life

# Run the optimization with more grid points
payoff_table = create_payoff_table(model)
results = augmecon2_optimization(model, payoff_table, grid_points=30)

####################################
####################################
# Run TOPSIS Analysis
def topsis_analysis(results, weights=None):
    """
    Implement TOPSIS method for finding the best compromise solution.
    
    Args:
        results (pandas.DataFrame): DataFrame containing 'actual_emission' and 'total_cost'
        weights (list, optional): Weights for criteria [cost_weight, emission_weight]. 
                                Default is equal weights [0.5, 0.5]
    
    Returns:
        dict: Dictionary containing TOPSIS results and analysis
    """
    # Extract decision matrix
    decision_matrix = np.array([
        results['total_cost'].values,
        results['actual_emission'].values
    ]).T
    
    # Set default weights if not provided
    if weights is None:
        weights = [0.5, 0.5]
    weights = np.array(weights)
    
    # Step 1: Normalize the decision matrix
    norm = np.sqrt(np.sum(decision_matrix**2, axis=0))
    normalized_matrix = decision_matrix / norm
    
    # Step 2: Calculate weighted normalized decision matrix
    weighted_normalized_matrix = normalized_matrix * weights
    
    # Step 3: Determine ideal and negative-ideal solutions
    ideal_solution = np.min(weighted_normalized_matrix, axis=0)  # Both cost and emissions are minimized
    negative_ideal_solution = np.max(weighted_normalized_matrix, axis=0)
    
    # Step 4: Calculate separation measures
    separation_ideal = np.sqrt(np.sum((weighted_normalized_matrix - ideal_solution)**2, axis=1))
    separation_negative = np.sqrt(np.sum((weighted_normalized_matrix - negative_ideal_solution)**2, axis=1))
    
    # Step 5: Calculate relative closeness to ideal solution
    relative_closeness = separation_negative / (separation_ideal + separation_negative)
    
    # Find the best compromise solution
    best_index = np.argmax(relative_closeness)
    
    # Calculate performance metrics
    emission_reduction = ((results['actual_emission'].max() - 
                          results['actual_emission'].iloc[best_index]) / 
                         results['actual_emission'].max() * 100)
    
    cost_increase = ((results['total_cost'].iloc[best_index] - 
                     results['total_cost'].min()) / 
                    results['total_cost'].min() * 100)
    
    # Store results
    topsis_results = {
        'best_solution': {
            'index': best_index,
            'emission': results['actual_emission'].iloc[best_index],
            'cost': results['total_cost'].iloc[best_index],
            'topsis_score': relative_closeness[best_index]
        },
        'performance': {
            'emission_reduction_percent': emission_reduction,
            'cost_increase_percent': cost_increase
        },
        'all_scores': relative_closeness
    }
    
    return topsis_results
topsis_results = topsis_analysis(results)