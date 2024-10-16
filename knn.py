import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors , KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
data=pd.read_csv('iris.csv')
print("Column headings:\n", data.columns.tolist())
p_column=input('Enter the column for prediction : ')
k=int(input('Enter choice for n_neighbour : '))
nk=1
if k%2==0:
    print('please choose between',k-1,' and ',k+1)
    nk=int(input())
else:
    nk=k
k=nk
if p_column=='sepal_length' or p_column=='sepal_width' or p_column=='petal_length' or p_column=='petal_width':
    M='Reg'
elif p_column=='species':
    M='Class'
else:
    print('Invalid choice')
  X=data.drop(columns=[p_column])
y= data[p_column]

if M == 'Class':
    model = KNeighborsClassifier(n_neighbors=k) 
elif M == 'Reg':
    model = KNeighborsRegressor(n_neighbors=k) 
else:
    print("Invalid option")
  model.fit(X,y)
X_new = np.array([[4.6, 3.6, 1, 0.2]]) 
y_pred = model.predict(X_new)
print(y_pred)
