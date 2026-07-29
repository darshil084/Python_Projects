import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


global_temp = pd.read_csv('GlobalTemperatures.csv')
'''print(global_temp.head())
print(global_temp.shape)
print(global_temp.columns)
print(global_temp.info())
print(global_temp.isnull().sum())'''

def wrangle(df):
    df = df.copy()
    df = df.drop(columns=['LandAverageTemperatureUncertainty', 'LandMaxTemperatureUncertainty', 
                            'LandMinTemperatureUncertainty', 'LandAndOceanAverageTemperatureUncertainty'], axis = 1)
    
    def converttemp(x):
        x = (x * 1.8) + 32
        return float(x)
    
    df["LandAverageTemperature"] = df["LandAverageTemperature"].apply(converttemp)
    df["LandMaxTemperature"] = df["LandMaxTemperature"].apply(converttemp)
    df["LandMinTemperature"] = df["LandMinTemperature"].apply(converttemp)
    df["LandAndOceanAverageTemperature"] = df["LandAndOceanAverageTemperature"].apply(converttemp)
    df["dt"] = pd.to_datetime(df["dt"])
    df["month"] = df["dt"].dt.month
    df["year"] = df["dt"].dt.year
    df = df.drop("dt", axis=1)
    df = df.drop("month", axis=1)
    df = df[df.year >= 1850]
    df = df.set_index(["year"])
    df = df.dropna()
    return df

global_temp = wrangle(global_temp)
#print(global_temp.head())

'''corrMatrix = global_temp.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()'''

target = 'LandAndOceanAverageTemperature'
y = global_temp[target]
x = global_temp[["LandAverageTemperature", "LandMaxTemperature", "LandMinTemperature"]]

xtrain, xval, ytrain, yval = train_test_split(x, y, train_size=0.25, random_state=42)
'''print(xtrain.shape)
print(xval.shape)
print(ytrain.shape)
print(yval.shape)'''

ypred = [ytrain.mean()] * len(yval)
print("Baseline MAE:", round(mean_absolute_error(yval, ypred), 5))


forest = make_pipeline(
    SelectKBest(k=2),
    StandardScaler(),
    RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=77,
        n_jobs=-1
    )
)

forest.fit(xtrain, ytrain)


rf_pred = forest.predict(xval)


rf_mae = mean_absolute_error(yval, rf_pred)
print("Random Forest MAE:", round(rf_mae, 5))
