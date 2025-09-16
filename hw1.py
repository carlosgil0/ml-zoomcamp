import pandas as pd
import numpy as np

cars = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv')

# Q1 pandas version
print(pd.__version__)

# Q2 Record counts
len(cars)

#q3 number of fuel types
print(cars.fuel_type.nunique())

# Q4 number of different columns with missing values
print((cars.isna().sum()>0).sum())

# Q5 max fuel efficiency in asia
print(cars[cars.origin == 'Asia'].fuel_efficiency_mpg.max())

# Q6 median hp after/before filling missing values
print(cars.horsepower.median())
cars_imp = cars.horsepower.fillna(cars.horsepower.mode().iloc[0])
print(cars_imp.median())

# q7 sum of weights of first 7 cars from asia w/only 2 features
X = cars[cars.origin == 'Asia'][['vehicle_weight', 'model_year']].head(7).values

XTX = X.T @ X 
XTX_inv = np.linalg.inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv @ X.T @ y
print(np.sum(w))