import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("Police_Department_Incidents_-_Previous_Year__2016_.csv")

print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())
print(pd.unique(data['Category']))

data.dropna(inplace=True,axis=0)
print(data.isnull().sum())

print(data.duplicated().sum())
data.drop_duplicates(inplace=True, keep=False)

print(data.shape)

data.Category.value_counts().nlargest(50).plot(kind='bar', figsize=(10,5))
plt.title('number of crimes category')
plt.show()

data.PdDistrict.value_counts().nlargest(50).plot(kind='bar', figsize=(10,5))
plt.title('number of crimes in districts')
plt.show()

data.Resolution.value_counts().nlargest(50).plot(kind='bar', figsize=(10,5))
plt.title('number of crimes resolutions')
plt.show()

data.DayOfWeek.value_counts().nlargest(50).plot(kind='bar', figsize=(10,5))
plt.title('number of crimes by day')
plt.show()

data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month

data.Month.value_counts().nlargest(50).plot(kind='bar', figsize=(10,5))
plt.title('number of crimes by month')
plt.show()

convert=['Category','Descript','DayOfWeek','Date','Time','PdDistrict','Resolution','Address','Location']
data[convert] = data[convert].astype('category')

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

for d in ['Category','Descript','DayOfWeek','Date','Time','PdDistrict','Resolution','Address','Location']: data[d] = label_encoder.fit_transform(data[d])

plt.figure(figsize=(10,5))
corr = data.corr()
sns.heatmap(corr, cmap="YlGnBu", annot=True )
plt.show()

x = data[['Category','Location','Address','X','Y']].values
y = data['PdDistrict'].values

print(x.shape)
print(y.shape)

from sklearn.preprocessing import StandardScaler
ssx=StandardScaler()
ssy=StandardScaler()

x=ssx.fit_transform(x)
y=ssy.fit_transform(y.reshape(-1,1))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 0)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.ensemble import RandomForestRegressor
rf_model=RandomForestRegressor(n_estimators= 20, random_state= 0)
rf_model.fit(x_train,np.ravel(y_train))

print('Random forest regresion model')
print(rf_model.score(x_train,y_train))
print(rf_model.score(x_test,y_test))

y_pred=rf_model.predict(x_test)
print('Y predicted values:',y_pred)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

plt.figure(figsize=(6, 4))
plt.scatter(y_test,y_pred)
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()

from sklearn.linear_model import LinearRegression
rl_model=LinearRegression()
rl_model.fit(x_train,np.ravel(y_train))

print('Linear regresion model')
print(rl_model.score(x_train,y_train))
print(rl_model.score(x_test,y_test))

y_pred=rl_model.predict(x_test)
print('Y predicted values:',y_pred)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

plt.figure(figsize=(6, 4))
plt.scatter(y_test,y_pred)
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()

from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(max_depth=4)
dt_model.fit(x_train, y_train)

print('Decision tree regresion model')
print(dt_model.score(x_train,y_train))
print(dt_model.score(x_test,y_test))

y_pred=dt_model.predict(x_test)
print('Y predicted values:',y_pred)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

plt.figure(figsize=(6, 4))
plt.scatter(y_test,y_pred)
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(data['X'],data['Y'], s = 1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('making a map of San Francisco by locations of crimes')
plt.show()

