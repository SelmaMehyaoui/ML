import numpy as np # linear algebra
import pandas as pd # data processing
import plotly as pl
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Input data files are available in the read-only "../input/" directory on Kaggle
Transfusion=pd.read_csv('/kaggle/input/blood-transfusion-dataset/transfusion.csv')
Covid=pd.read_csv('../input/covid-19-coronavirus-pandemic-dataset/Covid Live.csv')
Diabete=pd.read_csv('../input/diabetes-dataset/diabetes.csv')


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

import plotly.express as px
px.scatter(x=val_X.Glucose, y=val_predictions)
px.scatter(x=val_X.Insulin, y=val_predictions)

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

#TRAITER VIA PIPELINE AVEC CROSS VALIDATION AVEC PREPROCESSING (NUM ET CAT) in PIPELINE
#COMPARER RESULTATS POST CV POUR DIFFERENTES VALEURS (nombre d arbres) DU RANDOMFOREST

val_X.head()

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)


# Select categorical columns with relatively low cardinality 
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
