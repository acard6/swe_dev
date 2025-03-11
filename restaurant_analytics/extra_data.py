from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from meteostat import Point, Daily, units
import meteostat.units as units
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd
import holidays
import os

def get_weather():
    """ gets weather data using meteostat """
    start = datetime(2024, 3, 22)
    end = datetime(2025,3,9)

    city = Point(41.845044, -87.928607)

    data = Daily(city, start, end)
    data = data.convert(units.imperial)
    data = data.fetch()


    to_add = ["tavg", "prcp", "snow", "wspd"]
    df = data[to_add]


def add_data(df):
    """ writing dataframe data to excel data file 
        much thanks to stackpverflow 
        https://stackoverflow.com/questions/38074678/append-existing-excel-sheet-with-new-dataframe-using-python-pandas/38075046#38075046
    """
    cd = os.path.dirname(os.path.abspath(__file__))
    file_name = cd+"\\data.xlsx"
    file = os.path.join(cd, file_name)

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
    pass



def main():
    pass

if __name__ == "__main__":
	main()