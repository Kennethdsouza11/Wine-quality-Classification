import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Data Collection
#loading the data set into the pandas data frame
wine_dataset = pd.read_csv('winequality.csv')

#number of rows and columns in the dataset
print(wine_dataset.shape)

#to print the first 5 rows of the dataframe
print(wine_dataset.head())

#to print the last 5 rows of the dataframe
print(wine_dataset.tail())

#checking for missing values in the dataset
print(wine_dataset.isnull().sum())

#data analysis and visualization 
#statistical measures of the dataset
print(wine_dataset.describe())

#number of values for each quality
sns.catplot(x = 'quality',data = wine_dataset,kind='count')
#plt.show()

#volatile acidity vs quality
plt.figure(figsize = (5,5))
sns.barplot(x = 'quality',y='volatile acidity',data = wine_dataset)
#plt.show()

#citric acid content vs quality
sns.barplot(x = 'quality',y='citric acid',data = wine_dataset)
#plt.show()

#correlation
#postive correlation -- when one value increases and other value increases or vice versa
#negative correlation -- when one value increases and other value decrease or vice versa
correlation = wine_dataset.corr()
#constructing heatmap to understand the correlation between the columns
plt.figure(figsize = (10,10))
sns.heatmap(correlation, cbar=True,square=True,fmt='.1f', annot = True, annot_kws={'size':8}, cmap = 'Blues') #fmt is the precission of the values within the square, annot is the various description written beside the squares, annot_kws is the font size of those description, cbar is the color bar at the right hand side, cmap is the color of the squares here it is blue
# plt.show() #darker the color of the square (which is close to 1) it means that particular square is directly proprtional to the change and lighter the color means that square is inversely proprtional to the change. ignore the diagonal element as they belong to the same column. So always look into the other columns

#Data Preprocessing
#seperate the data and label
X = wine_dataset.drop('quality',axis = 1) #if we're dropping a columnn we need to mention that the axis is 1 if its row then we need to mention that the axis is equal to 0
print(X)

#label binraisation which means separiting this particular label into 2 values good and bad
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0) # lambda is usefull when we want to binarize the data or particular column into 2 values (here good and bad). lambda is a function

print(Y)

#split the data into training and test data
#train the model with the training data and test the model with the test data
#X_train data are more like the questions of the training set and the Y_train are the answers to the particular X_train quesion. Similarly for the X_test and the Y_test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 2)
#test_size = 0.2 means 20% is the test data set and 80% is the training data set
#label for X_train is stored in Y_train and the label for Y_train is stored in Y_test

print(Y.shape, Y_train.shape, Y_test.shape)#it will give the result in terms of values

#Random Forest Classifier
model  = RandomForestClassifier()
model.fit(X_train,Y_train) #fit function fits the data into our model

#evaluate our model
#accuracy score model 
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test) #here X_test_prediction are the values predicted by the trained model and Y_test contains the actual values

print('Accuracy : ',test_data_accuracy)

#building a predictive system
input_data = (7.9,0.6,0.06,1.6,0.069,15.0,59.0,0.9964,3.3,0.46,9.4)
#changing the input data into numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # rather than the model to look into thousands of values we tell the model that we are looking for only one particular value

prediction = model.predict(input_data_reshaped)

print(prediction)
#since the prediction is a list we are mentioning the first variable in the list
if (prediction[0] == 1):
    print("Wine quality is good!")
else:
    print("Wine quality is bad!")
