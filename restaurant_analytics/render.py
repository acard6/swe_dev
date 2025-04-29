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
		if df['count'][i] > 0:
			month = df["date"][i].month-1
			day = (df["idx"][i]+3)%7
			month_day[month][day][0] += df["count"][i]
			month_day[month][day][1] += 1
	return month_day


def print_weekly_data(input):
	print("weekly data")
	for i in range(len(input)):
		print("week {}: {}".format(i, input[i]))
	img_pth = os.path.join(cd,"images/weekly_covers.png")
	plt.plot(input, label="weekly covers")
	plt.xlabel("week")
	plt.ylabel("covers")
	plt.title("Weekly covers")
	plt.grid(True)
	plt.savefig(img_pth)
	plt.show()

def print_daily_data(input, selected_day='Sunday'):
	'''plotting a single days values'''
	values = input[selected_day]
	plt.plot(values, color="black", label=selected_day)
	img_pth = os.path.join(cd,f"images/{selected_day}_covers.png")
	# Customize the plot
	plt.xlabel("week")
	plt.ylabel("covers")
	plt.title(f"Values for {selected_day}")
	plt.grid(True)
	plt.savefig(img_pth)
	# Show the plot
	plt.show()

def print_daily_all(input):

#adding all days of the weeks data to a single graph
	colors = {
		"Monday": "red",
		"Tuesday": "blue",
		"Wednesday": "green",
		"Thursday": "purple",
		"Friday": "orange",
		"Saturday": "cyan",
		"Sunday": "magenta"
	}
	fig, ax = plt.subplots(figsize=(10, 5))

	# Plot each day's values on the same axis
	for (day, values) in input.items():
		x = range(1, len(values) + 1)  # X-axis (e.g., 1, 2, 3 for each value)
		ax.plot(x, values, marker='o', linestyle='-', color=colors[day], alpha=0.7, label=day)
	img_pth = os.path.join(cd,"images/daily_covers.png")
	# Customize the plot
	ax.set_xlabel("week")
	ax.set_ylabel("Values")
	ax.set_title("data for every data over time")
	ax.legend(loc="upper right")  # Show legend
	ax.grid(True)
	plt.savefig(img_pth)
	plt.show()

def print_daily_average(input):
	'''adding daily averages to a bar graph'''
	img_pth = os.path.join(cd,"images/daily_average.png")
	plt.bar(range(len(input)), list(input.values()), align='center')
	plt.xticks(range(len(input)), list(input.keys()), rotation=45)
	plt.grid(True)
	plt.title("Average covers per Day")
	plt.savefig(img_pth)
	plt.show()

def print_monthly_average(input):
	'''adding monthly totals and averages to abar graph'''
	img_pth = os.path.join(cd,"images/monthly_average.png")
	fig, ax = plt.subplots()
	plt.bar(range(len(input)), list(input.values()), align='center')
	plt.xticks(range(len(input)), list(input.keys()), rotation=45)
	plt.grid(True)
	plt.title("Monthly average covers")
	plt.savefig(img_pth)
	plt.show()


def main():
	d = weekly_total()
	# print_weekly_data(d)

	A = {}
	a = days_of_week()
	# print("data per given day")
	for key, val in a.items():
		#daily average throughout the week
		out = np.sum(val)/len(val)
		A[key] = out
		print("{}: {:.2f}".format(key,out) )
	# print_daily_data(a, selected_day='Monday')	# prints a line graph for each given day
	# print_daily_all(a)	# prints all the days on the same graph
	# print_daily_average(A)	# prints bar graph of daily averages


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
	# 	print("{}: {:.2f}".format(key,B[key]))
	# print_monthly_average(B)		

	c = month_day_joint()
	#day-month combo that exceeds daily & month avg  
	if False:
		print("joint data for given month and day")
		for i in range(len(c)):	# for month
			for j in range(len(c[0])):	 # for day of week
				month = calendar.month_name[i+1]
				day = calendar.day_name[j]
				avg = c[i][j][0]/c[i][j][1]
				margin_1 = bool(avg/A[day] > 0.9)
				margin_2 = bool(avg/B[month] > 0.9)
				str = f"{month}-{day}"
				print( f"{str:<19}\t{avg:.2f}" ) # uncomment to print total joint data

				# if margin_1 and margin_2:			# uncomment to print joint data grater than averages
				# 	print( f"{str:<19}\t{avg:.2f}" )

				# if month == "December":						# uncomment to print joint data of certain month
				# 	print( f"{str:<19}\t{avg:.2f}" )


if __name__ == "__main__":
	main()