import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot


url ="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(url, header=None)

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(url, names = headers)

### Convert "?" to NaN
df.replace("?", np.nan, inplace = True)
###Evaluating for Missing Data
missing_data = df.isnull()

# Count missing values in each column(True=missing)
# for column in missing_data.columns.values.tolist():
#     print(column)
#     print (missing_data[column].value_counts())
#     print("")

### Deal with missing data

"""
"normalized-losses": 41 missing data, replace them with mean
"stroke": 4 missing data, replace them with mean
"bore": 4 missing data, replace them with mean
"horsepower": 2 missing data, replace them with mean
"peak-rpm": 2 missing data, replace them with mean
Replace by frequency:

"num-of-doors": 2 missing data, replace them with "four". 
    * Reason: 84% sedans is four doors. Since four doors is most frequent, it is most likely to 
Drop the whole row:

"price": 4 missing data, simply delete the whole row
    * Reason: price is what we want to predict. Any data entry without price data cannot be used for p
"""

avg_1 = df["normalized-losses"].astype("float").mean(axis=0)
df["normalized-losses"].replace(np.nan, avg_1, inplace = True)

avg_2 = df["stroke"].astype("float").mean(axis=0)
df["stroke"].replace(np.nan, avg_2, inplace = True)

avg_3 = df["bore"].astype("float").mean(axis=0)
df["bore"].replace(np.nan, avg_3, inplace = True)

avg_4 = df["horsepower"].astype("float").mean(axis=0)
df["horsepower"].replace(np.nan, avg_4, inplace = True)

avg_5 = df["peak-rpm"].astype("float").mean(axis=0)
df["peak-rpm"].replace(np.nan, avg_5, inplace = True)

# print(df['num-of-doors'].value_counts())
# print(df['num-of-doors'].value_counts().idxmax())
#replace the missing 'num-of-doors' values by the most frequent
df["num-of-doors"].replace(np.nan, "four", inplace = True)

# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace = True)

# reset index, because we droped two rows
df.reset_index(drop = True, inplace = True)

###Convert data types to proper format
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
###Data Standardization
# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df["highway-mpg"] = 235/df["highway-mpg"]
# rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={'highway-mpg':'highway-L/100km'}, inplace=True)
# print(df[["length","width","height"]].head())
###Data Normalization
# replace (origianl value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()
# print(df[["length","width","height"]].head())

###Binning
df["horsepower"]=df["horsepower"].astype(float, copy=True)
binwidth = (max(df["horsepower"])-min(df["horsepower"]))/4
bins = np.arange(min(df["horsepower"]), max(df["horsepower"]), binwidth)
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names,include_lowest=True )


"""
# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
# plt.pyplot.show()
"""

###ndicator variable (or dummy variable)

dummy_variable_1 = pd.get_dummies(df["fuel-type"])
print(df["fuel-type"])
dummy_variable_1.rename(columns={'diesel':'fuel-type-diesel', 'gas':'fuel-type-diesel'}, inplace=True)
# merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variable_1], axis=1)
# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)
print(df["fuel-type-diesel"])
# get indicator variables of aspiration and assign it to data frame "dummy_variable_2"
dummy_variable_2 = pd.get_dummies(df['aspiration'])
# change column names for clarity
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

df.drop("aspiration", axis = 1, inplace=True)

# df.to_csv('clean_df.csv')
print (df[['make', 'price']])