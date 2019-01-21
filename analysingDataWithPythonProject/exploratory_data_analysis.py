import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

path='./file/clean_df.csv'
df = pd.read_csv(path)


# Engine size as potential predictor variable of price
#1 Positive linear relationship
# df[["engine-size", "price"]].corr()
# sns.regplot(x="engine-size", y="price", data=df)
# plt.ylim(0,)
# plt.show()


#2 Negative linear relationship
# df[['highway-mpg', 'price']].corr()
# sns.regplot(x="highway-mpg", y="price", data=df)
# plt.show()

#3 Weak Linear Relationship
# df[['peak-rpm','price']].corr()
# sns.regplot(x="peak-rpm", y="price", data=df)
# plt.show()

###4 Categorical variables*(descriptive)
# sns.boxplot(x="body-style", y="price", data=df)
# sns.boxplot(x="engine-location", y="price", data=df)
# # drive-wheels
#4.3 sns.boxplot(x="drive-wheels", y="price", data=df)
# plt.show()
# Descriptive Statistical Analysis
#5 print(df.describe(include=['object']))

### Value Counts
# drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
# drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
# drive_wheels_counts.index.name = 'drive-wheels'
# print(drive_wheels_counts)

# engine-location as variable
# engine_loc_counts = df['engine-location'].value_counts().to_frame()
# engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
# engine_loc_counts.index.name = 'engine-location'
# engine_loc_counts.head(10)

### Basic of Grouping
# print(df['drive-wheels'].unique())
# df_group_one=df[['drive-wheels','b0ody-style','price']]
# df_group_one=df_group_one.groupby(['drive-wheels'],as_index= False).mean()


#6 grouping results
# df_gptest=df[['drive-wheels','body-style','price']]
# grouped_test1=df_gptest.groupby(['drive-wheels','body-style'],as_index= False).mean()
# grouped_pivot=grouped_test1.pivot(index='drive-wheels',columns='body-style')
# grouped_pivot=grouped_pivot.fillna(0) #fill missing values with 0
# print(grouped_pivot)
# fig, ax=plt.subplots()
# im=ax.pcolor(grouped_pivot, cmap='RdBu')
# # # #label names
# row_labels=grouped_pivot.columns.levels[1]
# col_labels=grouped_pivot.index
# #move ticks and labels to the center
# ax.set_xticks(np.arange(grouped_pivot.shape[1])+0.5, minor=False)
# ax.set_yticks(np.arange(grouped_pivot.shape[0])+0.5, minor=False)
# #insert labels
# ax.set_xticklabels(row_labels, minor=False)
# ax.set_yticklabels(col_labels, minor=False)
# #rotate label if too long
# plt.xticks(rotation=90)
#
# fig.colorbar(im)
# plt.show()

###7 Correlation and Causation

# Wheel-base vs Price
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("Wheel-base vs Price,The Pearson Correlation Coefficient is  ", pearson_coef, " with a P-value of P =", p_value)
#
# Horsepower vs Price
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("Horsepower vs Price,The Pearson Correlation Coefficient is  ", pearson_coef, " with a P-value of P =", p_value)

# Length vs Price
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("Length vs Price,The Pearson Correlation Coefficient is      ", pearson_coef, " with a P-value of  P =", p_value)

# Width vs Price
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("Width vs Price,The Pearson Correlation Coefficient is       ", pearson_coef, " with a P-value of P =", p_value )

# Curb-weight vs Price
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "Curb-weight vs Price, The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

### ANOVA
# grouped_test2=df_gptest[['drive-wheels','price']].groupby(['drive-wheels'])
# f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'],
#                               grouped_test2.get_group('4wd')['price'])
# print( "ANOVA results: F=", f_val, ", P =", p_val)

# Separately: fwd and rwd¶
# f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])

# 4wd and rwd
# f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])
# 4wd and fwd
# f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])
# print("ANOVA results: F=", f_val, ", P =", p_val)
# means(fwd=9244.77966101695, rwd=19757.613333333335,4wd=10241.0)
# print(grouped_test2.get_group('4wd')['price'].mean())