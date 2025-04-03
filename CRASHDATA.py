
import pandas as pd
dataFrame = pd.read_csv("C:\\Users\\kiran\\Desktop\\data\\crashdata.csv")
print("Our DataFrame....\n",dataFrame)
df = pd.read_csv("C:\\Users\\kiran\\Desktop\\data\\crashdata.csv")
## Display check the first five rows of the data to understand its structure
print(df.head(5))
# Check for missing values
print(df.isnull().sum())
# To get overall information 
print(df.info())
# Check for missing values in each column
print(df.isnull().sum())
# Drop rows where 'CRASH_DATE' is missing (if this is critical to the analysis)
df.dropna(subset=['CRASH_DATE'], inplace=True)
# You can fill missing values in other columns, for example, filling with 'Unknown'
df['WEATHER_CONDITION'].fillna('Unknown', inplace=True)

# Convert 'CRASH_DATE' to datetime format for easier analysis
df['CRASH_DATE'] = pd.to_datetime(df['CRASH_DATE'], errors='coerce')



import matplotlib.pyplot as plt
import seaborn as sns
# Count crashes by weather condition
weather_counts = df['WEATHER_CONDITION'].value_counts()
# Display the result of weather counts
print(weather_counts)
import pandas as pd
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='WEATHER_CONDITION', order=df['WEATHER_CONDITION'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Number of Crashes by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Crashes')
plt.show()


# Sum the total injuries by crash type
injuries_by_crash_type = df.groupby('CRASH_TYPE')['INJURIES_TOTAL'].sum()
# Display the result
print(injuries_by_crash_type)
plt.figure(figsize=(12,6))
sns.barplot(x=injuries_by_crash_type.index, y=injuries_by_crash_type.values)
plt.xticks(rotation=45)
plt.title('Total Injuries by Crash Type')
plt.ylabel('Total Injuries')
plt.show()


# Extract the month from the 'CRASH_DATE'
df['CRASH_DATE'] = pd.to_datetime(df['CRASH_DATE'], format='%m/%d/%Y %H:%M')
df['CRASH_MONTH'] = df['CRASH_DATE'].dt.month
crashes_by_month = df['CRASH_MONTH'].value_counts().sort_index()
print(crashes_by_month)
plt.figure(figsize=(12,6))
sns.barplot(x=crashes_by_month.index, y=crashes_by_month.values)
plt.title('Crashes by Month')
plt.ylabel('Number of Crashes')
plt.show()


import calendar
#These data are extracted from the avove crash_month 
data = {
    'CRASH_MONTH': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Counts': [66068, 65284, 67812, 66417, 77268, 77697, 78570, 80823, 82228, 86681, 78184, 77000]
}
df = pd.DataFrame(data)
df['Month_Name'] = df['CRASH_MONTH'].apply(lambda x: calendar.month_name[x])
print(df[['Month_Name', 'Counts']])
month_names = [calendar.month_name[i] for i in crashes_by_month.index]
plt.figure(figsize=(12,6))
sns.barplot(x=month_names, y=crashes_by_month.values)
plt.title('Crashes by Month')
plt.xlabel('Month')
plt.ylabel('Number of Crashes')
plt.show()



# To determine the 'CRASH_Hour' with datetime format
dataFrame = pd.read_csv("C:\\Users\\kiran\\Desktop\\data\\crashdata.csv")
print("Our DataFrame....\n",dataFrame)
df = pd.read_csv("C:\\Users\\kiran\\Desktop\\data\\crashdata.csv")
# Extract the Hour from the 'CRASH_DATE'
df['Hour'] = pd.to_datetime(df['Hour'], format='%H:%M')
df['Hour'] = df['Hour'].dt.hour
crashes_by_hour = df['Hour'].value_counts().sort_index()
print(crashes_by_hour)
plt.figure(figsize=(24,6))
sns.barplot(x=crashes_by_hour.index, y=crashes_by_hour.values)
plt.title('Crashes by Hour')
plt.ylabel('Number of Crashes')
plt.show()



# To determine the crashes Type
dataFrame = pd.read_csv("C:\\Users\\kiran\\Desktop\\data\\crashdata.csv")
print("Missing values in 'FIRST_CRASH_TYPE':", dataFrame['FIRST_CRASH_TYPE'].isnull().sum())
dataFrame['FIRST_CRASH_TYPE'] = dataFrame['FIRST_CRASH_TYPE'].fillna('Unknown')
crash_types = dataFrame['FIRST_CRASH_TYPE'].value_counts()
print("\nMost Frequent Crash Types:\n", crash_types)
plt.figure(figsize=(20, 6))
sns.barplot(x=crash_types.index, y=crash_types.values, palette='Set2') 
plt.title('Most Frequent Crash Types')
plt.xlabel('Crash Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# Traffic Control Signals Crash
dataFrame = pd.read_csv("C:\\Users\\kiran\\Desktop\\data\\crashdata.csv")
print("Missing values in 'TRAFFIC_CONTROL_DEVICE':", dataFrame['TRAFFIC_CONTROL_DEVICE'].isnull().sum())
dataFrame['TRAFFIC_CONTROL_DEVICE'] = dataFrame['TRAFFIC_CONTROL_DEVICE'].fillna('Unknown')
device_counts = dataFrame['TRAFFIC_CONTROL_DEVICE'].value_counts()
print("\nTraffic Control Device Counts:\n", device_counts)
plt.figure(figsize=(10, 6))
sns.barplot(x=device_counts.index, y=device_counts.values, palette='Set3')
plt.title('Distribution of Traffic Control Devices')
plt.xlabel('Traffic Control Device')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#Severity Breakdown by Traffic Control Devices
severity_by_device = dataFrame.groupby('TRAFFIC_CONTROL_DEVICE')['MOST_SEVERE_INJURY'].value_counts().unstack()
print("\nCrash Severity Breakdown by Traffic Control Device:\n", severity_by_device)
plt.figure(figsize=(12, 8))
severity_by_device.plot(kind='bar', stacked=True, colormap='Set2')
plt.title('Crash Severity Breakdown by Traffic Control Device')
plt.xlabel('Traffic Control Device')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Most Severe Injury', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

