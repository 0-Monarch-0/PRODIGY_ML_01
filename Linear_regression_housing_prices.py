#importing required data 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

#getting data training and testing files 
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

selected_columns = ['TotalBsmtSF', 'FullBath', 'BedroomAbvGr']

#selected features to pridict housing values.
X_train = train_df[selected_columns]
y_train = train_df['SalePrice'] #target values.

X_test = test_df[selected_columns]
y_test = sample_sub['SalePrice']

#X_test.isnull().sum()#total bsmt have a null value so we fill that with mean of X
#y_test.isnull().sum()
X_test.fillna(X_test.mean(), inplace=True)

# now we scale the attributes so they work better , and we are doing that with Standard Scaler as we have
#  bigger values in Square foot and smaller values in no of rooms.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# now we get to train the model
model = LinearRegression()
model.fit(X_train_scaled,y_train)

#predict on test set
y_pred = model.predict(X_test_scaled)

#evaluate the model
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test,y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r_squared}")
#Mean Squared Error: 2989299292.7141595
#R-squared: -9.963174028763461
# a high mean squared error means we are far away from the acutal values. so the features we selected are not appropriate. 
# and from the actual data containing 33 columns we only formed this prediction model using 3. 