from datetime import datetime
from meteostat import Point, Daily, units, Stations
import meteostat.units as units
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd
import holidays
import os
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression


cd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_name = cd+"\\datas\predictions.xlsx"
file = os.path.join(cd, file_name)


def get_weather(start, end):
    """ gets weather data using meteostat 
        input Datetime(yyyy,mm,dd)
    """
    start = start
    end = end
    # end = datetime(2025,3,9)

    # station = Stations()
    # station = station.nearby(41.845044, -87.928607)
    # station = station.inventory('daily', (start,end))
    # station.fetch(5)
    # print(station)

    city = Point(41.845044, -87.928607)

    # data = Daily(72534, start, end)
    data = Daily(city, start, end)
    data = data.convert(units.imperial)
    data = data.fetch()


    to_add = ["tavg", "prcp", "snow", "wspd"]
    df = data[to_add]
    return  df


def add_data(df):
    """ writing dataframe data to excel data file 
        much thanks to stackpverflow 
        https://stackoverflow.com/questions/38074678/append-existing-excel-sheet-with-new-dataframe-using-python-pandas/38075046#38075046
    """
    if os.path.isfile(file):  # if file already exists append to existing file
        workbook = openpyxl.load_workbook(file)  # load workbook if already exists
        sheet = workbook['Sheet1']  # declare the active sheet 

        # append the dataframe results to the current excel file
        for row in dataframe_to_rows(df, header = False, index = False):
            sheet.append(row)
        workbook.save(file)  # save workbook
        workbook.close()  # close workbook
    else:  # create the excel file if doesn't already exist
        with pd.ExcelWriter(path = file, engine = 'openpyxl') as writer:
            df.to_excel(writer, index = False, sheet_name = 'Sheet1')

    print("data added")


def get_holidays():
    df = pd.read_excel(file, usecols="C")
    # time = df["date"][104] # independance day
    us_holidays = holidays.US(categories=("public","unofficial"))
    bank_holidays = holidays.NYSE()
    for date in df["date"]:
        text = ""
        us = us_holidays.get(date) 
        bank = bank_holidays.get(date)
        if bank == None and us == None:
            continue
        elif us == bank:
            text = us
        elif bank == None:
            text = us
        elif us == None:
            text = bank
        else:
            text =us

        print(text,end="\t")
        print("{}-{}-{}".format(date.year, date.month, date.day) )

def analsis():
    x = [25.73, 14.34, 14.66, 14.48, 14.21, 14.57, 25.91, 14.10, 14.19, 14.00]
    y = [20.24, 17.54, 16.21, 22.45, 16.76, 15.73, 20.15, 16.02, 16.12, 17.32]
    z = [14.63, 39.02, 46.34, 36.59, 43.90, 46.34, 17.07, 48.78, 46.34, 53.66]

    data = np.column_stack((x,y,z))
    pap = PCA()
    pap.fit(data)

    print("Explained Variance Ratio:", pap.explained_variance_ratio_)
    print("Principal Components (directions):\n", pap.components_)
    if pap.explained_variance_ratio_[1] > 0.2:
        pc12(x,y,z)
    else:
        pc1(data,pap,x,y,z)

def pc1(data, pap,x,y,z):

    X = np.column_stack((x, y))

    # Fit model
    reg = LinearRegression().fit(X, z)

    # Get coefficients
    a, b = reg.coef_
    c = reg.intercept_

    print(f"z = {a:.3f}·x + {b:.3f}·y + {c:.3f}")

    center = pap.mean_
    pc1_dir = pap.components_[0]
    pc1_line = np.outer(np.linspace(-3, 3, 100), pc1_dir) + center  # extend line in both directions

    # Project data onto PC1 line
    projected = pap.transform(data)[:, 0:1]  # keep only PC1
    reconstructed = pap.inverse_transform(np.hstack((projected, np.zeros((len(data), 2)))))

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Original data
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='blue', label='Original Data')

    # PC1 direction line
    ax.plot(pc1_line[:, 0], pc1_line[:, 1], pc1_line[:, 2], color='red', label='PC1 Line')

    # Projections (dots projected onto line)
    ax.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2],
            color='green', label='Projected Points')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('PCA Visualization: Data & Principal Component')
    ax.legend()
    plt.show()


def pc12(x,y,z):
    X = np.column_stack((x, y))

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, z)

    # Get coefficients and intercept
    a, b = model.coef_
    c = model.intercept_
    print(f"Best fit plane: z = {a:.3f}·x + {b:.3f}·y + {c:.3f}")
    z_pred = model.predict(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original data
    ax.scatter(x, y, z, color='blue', label='Original Data')

    # Plot predicted (fitted) line
    ax.plot(x, y, z_pred, color='red', label='Best Fit Line (Reg)')

    # Or optionally: make a surface
    # Make a grid for better visualization
    x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 20),
                                np.linspace(min(y), max(y), 20))
    z_grid = model.predict(np.column_stack((x_grid.ravel(), y_grid.ravel()))).reshape(x_grid.shape)

    # Plot the plane
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Linear Regression Best Fit Line in 3D')
    plt.show()


def fourier_anal_sis():
    pass

def main():
    # get_holidays()
   
    
    start = datetime(2025, 10, 21)
    end = datetime(2025,11, 5)
    df = get_weather(start, end)
    print(df)
    
    
    # analsis()

if __name__ == "__main__":
	main()

'''
WEEKEND DATA:
32x4        avg acc - 25.293%
training:   24.91, 21.51, 21.84, 20.80, 20.50, 23.00, 20.60, 22.42, 19.63, 23.95
testing:    34.20, 25.17, 30.59, 32.08, 34.37, 28.33, 31.12, 32.80, 32.46, 28.88
Accuracy:   11.76, 35.29, 35.29, 23.53, 23.53, 35.29, 23.53, 23.53, 23.53, 17.65 - d
        std div - 7.866%
        highest - 35.29%
        lowest - 11.76%
        median - 23.53%

64x32       avg acc - 6.468%
training:   17.80, 17.99, 18.02, 18.22, 18.37, 17.99, 18.00, 18.48, 18.23, 18.71
testing:    41.70, 36.55, 35.46, 34.99, 34.67, 36.94, 39.15, 40.49, 36.25, 38.98
Accuracy:   00.00, 11.76, 00.00, 5.88,  11.76, 5.88,  0.00,  11.76, 5.88,  11.76 - c
        std div - 5.148%
        highest - 11.76%
        lowest - 0.0%
        median - 5.88%

WEEKDAY DATA
64x32       avg acc - 40.974%
training:   12.69, 13.55, 14.40, 14.41, 13.95, 14.12, 14.00, 13.79, 13.97, 14.25
testing:    25.33, 20.53, 16.82, 16.70, 18.82, 18.64, 17.20, 18.90, 19.97, 16.44
Accuracy:   43.90, 41.46, 36.59, 41.46, 41.46, 46.34, 36.59, 39.02, 39.02, 43.90 - b
        std div - 3.209%
        highest - 46.34%
        lowest - 36.59%
        median - 41.46%

32x4        avg acc - 39.267%
training:   25.73, 14.34, 14.66, 14.48, 14.21, 14.57, 25.91, 14.10, 14.19, 14.00
testing:    20.24, 17.54, 16.21, 22.45, 16.76, 15.73, 20.15, 16.02, 16.12, 17.32
Accuracy:   14.63, 39.02, 46.34, 36.59, 43.90, 46.34, 17.07, 48.78, 46.34, 53.66 - a
        std div - 13.233%
        highest - 53.66%
        lowest - 14.63%
        median - 46.34%

TOTAL
average - 28.001%
std div - 16.114%
highest - 53.66%
lowest - 0.0%
median - 35.29%
'''

'''
    ALL:
Explained Variance Ratio: [0.92929517 0.04370666 0.02699817]
Principal Components (directions):
 [[ 0.44302331  0.12716341 -0.88744567]
 [-0.89153764  0.16660444 -0.42119307]
 [ 0.09429204  0.97778956  0.18718062]]
z = -1.422·x + -0.814·y + 79.815

    Weekend 32x4
Explained Variance Ratio: [0.90310887 0.07347322 0.02341791]
Principal Components (directions):
 [[ 0.06683018  0.24224916 -0.96790964]
 [ 0.46101105 -0.86781785 -0.18536719]
 [ 0.88487431  0.43382891  0.16967596]]
z = -1.893·x + -1.855·y + 124.297

    Weekend 64x32
Explained Variance Ratio: [0.82442526 0.17443009 0.00114465]
Principal Components (directions):
 [[-0.03785764  0.09767485 -0.99449808]
 [ 0.01171339  0.9951865   0.09729657]
 [-0.99921449  0.00796552  0.03881952]]
z = 13.163·x + -0.275·y + -222.525


    Weekday 64x32
Explained Variance Ratio: [0.67748002 0.32136644 0.00115354]
Principal Components (directions):
 [[-0.08609232  0.50066591  0.8613488 ]
 [-0.16131584  0.84614518 -0.50795229]
 [ 0.98314054  0.18267999 -0.00791874]]
Best fit plane: z = 3.382·x + 1.001·y + -25.044

    Weekday 32x4
Explained Variance Ratio: [0.9743399  0.01917612 0.00648399]
Principal Components (directions):
 [[ 0.32696803  0.11319084 -0.93823224]
 [-0.66273294  0.73521922 -0.14225945]
 [ 0.67370391  0.66831171  0.31540847]]
z = -2.160·x + -1.557·y + 102.972
'''