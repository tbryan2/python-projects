import pandas as pd

from matplotlib import pyplot as plt

data = pd.read_csv("data/coffees.csv")

#part 1 (viewing and cleaning data):

#print(data) for a print of all the data

#print(data.head()) for the first few rows of data set starting at 0

#print(data.describe(include="all")) to get a description of the all data (data count, unique values, tip, frequency

#print(data.loc[2]) to print the data from the 3rd set (or 2 because it starts at 0)

#print(data.coffees) to print a specific column

##print(data.coffees[:5]) to print the first 5 values in a specific column

#print(len(data)) to get the length of the dataset

#print(data.isnull()) to get the null data (true/false)

#print(data.coffees.isnull()) to get the null data from the coffees column

#print(data[data.coffees.isnull()]) to get the data in the other columns where the data in coffees is null

#print(data.timestamp[0]) to get the timestamp of the first value

#print(type(data.timestamp[0])) to get the type of data of the first value

data.coffees = pd.to_numeric(data.coffees, errors="coerce")

#converts an non-numeric values to numeric values, errors="coerce" turns any strings into NaNs

data = data.dropna() #redefines data as the data with the NaNs dropped

data.coffees = data.coffees.astype(int) #takes the coffees column and changes all the values to integers, must define as numeric first

data.timestamp = pd.to_datetime(data.timestamp) #converts the data in timestamp to timestamp data (sounds redundant but must be defined)

#part 2 (analysis of the data):

#data.coffees.plot() #defines the plot as the coffees aganist the index (default)

#data.plot('timestamp', style=".-") #plot the data with timestamp as the x values instead of the index, in the line style .-

#print(data.timestamp < "2013-03-01") #make all the data before this date true and all the data after this date false

#print(data[data.timestamp < "2013-03-01"]) #print all the data in every column before this date

data = data[data.timestamp < "2013-03-01"] #the data is now equal to the data where the timestamp is before this date

#data.plot('timestamp', style=".-", figsize=(15,4)) #replotting the coffees versus timestamp but with only the correct dates and figsize changes the size of the plot window

#print(data.contributor) #printing the contributors column

#print(data.contributor.value_counts()) #counts the amount of coffee each contributor contributed to the coffeees column

weekdays = data.timestamp.dt.weekday #creating a variable that equals the weekdays

data = data.assign(weekdays=weekdays) #creating a new column of using the variable weekdays, by default this is an index starting at 0

weekday_names =["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] #defining a variable of weekday names

weekday_dict = {key: weekday_names[key] for key in range(7)} #creating a dictionary of weekday_names

def day_of_week(idx):
    return weekday_dict[idx]

data.weekdays = data.weekdays.apply(day_of_week) #these past few lines are used to change the index to whatever column name and value you want

weekday_counts = data.groupby("weekdays").count() #group the columns together by weekday and count them

weekday_counts = weekday_counts.loc[weekday_names] #order the weekday column by the list weekday_names defined earlier

#weekday_counts.coffees.plot(kind="bar") #output a bar graph of the weekday_counts value of only the coffees column

#data.contributor.value_counts().plot(kind="bar") #outputs a bar graph of coffees per person 

data.index = data.timestamp #redefine the index as the timestamp column

data.drop(["timestamp"], axis=1, inplace=True) #drop the timestamp column since it is now the index

midnights = pd.date_range(data.index[0], data.index[-1], freq="D", normalize=True) #create a variable that is a date range starting in the first row, 0, and ending in the last row, -1, with a daily frequency, D, and normalized

new_index = midnights.union(data.index) #come up with a new_index that has the value of midnights and the original index in one

upsampled_data = data.reindex(new_index) #change the index to new_index

upsampled_data = upsampled_data.interpolate(method="time") #interpolate the number of coffees between each day

daily_data = upsampled_data.resample("D").asfreq() #daily_data is defined as the upsampled data resampled daily

daily_data = daily_data.drop(["contributor"], axis=1) #daily_data drops the column (axis=1) of contributors

daily_data["weekdays"] = daily_data.index.weekday_name #take the weekdays column and set it as the index.weekday_name

#daily_data.plot(style=".-") #plotting the daily_data variable

coffees_made = daily_data.coffees.diff().shift(-1) #cofees made is the daily data coffees column and is the difference between each row and shift the data up one

daily_data["coffees_made_today"] = coffees_made #add this variable as a column

coffees_by_day = daily_data.groupby("weekdays").mean() #create a variable coffees by day that is the daily data grouped by weekday and averaged out

coffees_by_day = coffees_by_day.loc[weekday_names] #order the coffees by day by the weekdays

#coffees_by_day.coffees_made_today.plot(kind="bar") #plot the bar graph of the average coffees per weekday

people = pd.read_csv("data/department_members.csv", index_col="date", parse_dates=True) #people is the data retrieved from this csv file

daily_data = daily_data.join(people, how="outer").interpolate(method="nearest") #join daily data and the people columns with the interpolation nearest method

daily_data = daily_data.interpolate(method="nearest")

daily_data["coffees_per_person"] = daily_data.coffees_made_today / daily_data.members

#daily_data.coffees_per_person.plot() #plot of coffees per person

machine_status = pd.read_csv("data/coffee_status.csv", index_col="date", parse_dates=True)

numerical_status = machine_status.status == "OK"

daily_data = daily_data.join(machine_status)

daily_data["numerical_status"] = daily_data.status == "OK"

daily_data[["numerical_status", "coffees_made_today"]].plot()

plt.show() #this is like the print function but with plots




