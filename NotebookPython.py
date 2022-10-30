import numpy as np # linear algebra
import pandas as pd # data processing
import plotly as pl
# Input data files are available in the read-only "../input/" directory on Kaggle

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

Transfusion=pd.read_csv('/kaggle/input/blood-transfusion-dataset/transfusion.csv')
Covid=pd.read_csv('../input/covid-19-coronavirus-pandemic-dataset/Covid Live.csv')
Diabete=pd.read_csv('../input/diabetes-dataset/diabetes.csv')

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# Create target object and call it y
y = Diabete.Outcome
# Create X
features = ['Glucose','Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'Age']
X = Diabete[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
diabete_model = DecisionTreeRegressor(random_state=1)
# Fit Model
diabete_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = diabete_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)

#confusion_matrix(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

#TROUVER MAX LEAF NODES
leaf_depth=[25, 50, 100, 150, 200]
MAE_tab=[]
for k in range(len(leaf_depth)):
    diabete_model=DecisionTreeRegressor(max_leaf_nodes=leaf_depth[k], random_state=1)
    diabete_model.fit(train_X, train_y)
    val_predictions=diabete_model.predict(val_X)
    MAE_tab.append(mean_absolute_error(val_predictions, val_y))
index_mae=MAE_tab.index(min(MAE_tab))  
best_leaf_depth=leaf_depth[index_mae]
print(best_leaf_depth)  


# Using best value for max_leaf_nodes
diabete_model = DecisionTreeRegressor(max_leaf_nodes=best_leaf_depth, random_state=1)
diabete_model.fit(train_X, train_y)
val_predictions = diabete_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
#confusion_matrix(val_predictions, val_y)
#print(confusion_matrix)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))


#RANDOM FOREST
# Specify Model
diabete_model = RandomForestClassifier(random_state=1)
# Fit Model
diabete_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = diabete_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print(val_predictions)
print(val_y)
print("Validation MAE : {:,.0f}".format(val_mae))

A=confusion_matrix(val_predictions, val_y)
print(A)
a=A[0][1]
b=A[1][0]
mae_homemade=(a+b)/sum(sum(A))
print(mae_homemade)

Ndiabetes=sum(Diabete.Outcome)
print(Ndiabetes)
Ntotal=Diabete['Outcome'].size
print(Ntotal)
RatioDiabetes=Ndiabetes/Ntotal
PercentDiabetes=RatioDiabetes*100
print('Le pourcentage de diabètes vaut : ') 
print(PercentDiabetes)

from sklearn.ensemble import RandomForestRegressor

# COMPARING RANDOM FOREST ACCURACIES

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=train_X, X_v=val_X, y_t=train_y, y_v=val_y):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)
    
# Fill in the best model
mae_table=[]
for i in range(0, len(models)):
    mae_table.append(score_model(models[i]))
    print("Model %d MAE: %d" % (i+1, mae_table[i]))

best_model = models[mae_table.index(min(mae_table))]
print(models.index(best_model))
