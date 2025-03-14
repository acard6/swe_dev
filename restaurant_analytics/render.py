import pandas as pd
import numpy as np
import os
import calendar
import json
import matplotlib.pyplot as plt

cd = os.path.dirname(os.path.abspath(__file__))
file_name = cd+"\\covers_data.xlsx"
fp = os.path.join(cd, file_name)


def weekly_total():
	""" returns weekly total as an array(based of off first idx and not by days of the week)"""
	df = pd.read_excel(fp, usecols="B,D",)
	week = 0
	l = len(df["count"])
	ret = []
	for i in range (l):
		if (df["idx"][i])%7 == 0:
			week += df["count"][i]
			ret.append(week)
			week = 0
			continue
		week += df["count"][i]
	ret.append(week)
	ret_np = np.array(ret)
	return ret_np

def monthly_total():
	"""returns a dict of monthly total. {april:X, may:Y, ...} """
	df = pd.read_excel(fp, usecols="C:D",)
	l = len(df["count"])
	ret = {}
	month = df["date"][0].month
	tot = 0
	for i in range(l):
		if df["date"][i].month != month:
			ret[calendar.month_name[month]] = tot
			month = df["date"][i].month
			# print(tot)
			tot = 0
		tot += df["count"][i]

	ret[calendar.month_name[month]] += tot
	return ret

def days_of_week():
	"""dict of array for a given day of the week(i.e. 'Sunday:[1st sunday, 2nd, ..]')"""
	df = pd.read_excel(fp, usecols="A,D",)
	l = len(df["count"])
	ret = {"Sunday":[], "Monday":[], "Tuesday":[], "Wednesday":[], "Thursday":[], "Friday":[], "Saturday":[]}
	for i in range(l):
		ret[df["day"][i]].append(df["count"][i])

	for key in ret.keys():
		ret[key] = np.array(ret[key])
	return ret

def month_day_joint():
	"""average count of a given day for a certain month(avg of wednesdays in september)"""
	df = pd.read_excel(fp, usecols="A:D",)
	month_day =np.zeros(shape=(12,7,2))
	l = len(df["count"])
	for i in range(l):
		month = df["date"][i].month-1
		day = (df["idx"][i]+3)%7
		month_day[month][day][0] += df["count"][i]
		month_day[month][day][1] += 1
	return month_day


def main():
	d = weekly_total()
	# print("weekly data")
	# for i in range(len(d)):
	# 	print("week {}: {}".format(i, d[i]))
	# img_pth = os.path.join(cd,"images/weekly_covers.png")
	# plt.plot(d, label="weekly covers")
	# plt.xlabel("week")
	# plt.ylabel("covers")
	# plt.title("Weekly covers")
	# plt.grid(True)
	# plt.savefig(img_pth)
	# plt.show()


	A = {}
	a = days_of_week()
	# print("data per given day")
	for key, val in a.items():
		#daily average throughout the week
		out = np.sum(val)/len(val)
		A[key] = out
		# print("{}: {:.2f}".format(key,out) )
	img_pth = os.path.join(cd,"images/daily_average.png")
	fig, ax = plt.subplots()
	plt.bar(range(len(A)), list(A.values()), align='center')
	plt.xticks(range(len(A)), list(A.keys()), rotation=45)
	plt.grid(True)
	plt.title("Average covers per Day")
	plt.savefig(img_pth)
	plt.show()


	# print("\ndata per given month")
	B = {}
	b = monthly_total()
	for key, val in b.items():
		#daily average for each month
		if key == "February":
			B[key] = val/28
		elif key == "April" or key == "June" or key == "September" or key == "November":
			B[key] = val/30
		else:
			B[key] = val/31 
		# print("{}: {:.2f}".format(key,B[key]))
	# img_pth = os.path.join(cd,"images/monthly_average.png")
	# fig, ax = plt.subplots()
	# plt.bar(range(len(B)), list(B.values()), align='center')
	# plt.xticks(range(len(B)), list(B.keys()), rotation=45)
	# plt.grid(True)
	# plt.title("Monthly average covers")
	# plt.savefig(img_pth)
	# plt.show()
		

	# print("\njoint data for given month and day")
	c = month_day_joint()
	#day-month combo that exceeds daily & month avg  
	for i in range(len(c)):
		for j in range(len(c[0])):
			month = calendar.month_name[i+1]
			day = calendar.day_name[j]
			avg = c[i][j][0]/c[i][j][1]
			margin_1 = bool(avg/A[day] > 0.9)
			margin_2 = bool(avg/B[month] > 0.9)

			# if margin_1 and margin_2 and day:
			# 	print( "{}:{}:{}".format(month, day, avg) )
			# if month == "April":
			# print( "{}-{}:{:.2f}".format(month, day, avg) )


if __name__ == "__main__":
	main()