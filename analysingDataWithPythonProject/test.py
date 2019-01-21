import pandas as pd
import numpy as np
path ="./test.csv"
df = pd.read_csv(path, header=None)
headers = ["fuel"]
df.columns = headers
pd.get_dummies(df["fuel"])
print(df["fuel"])