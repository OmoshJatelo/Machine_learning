
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
hours=np.array([1,2,3,4,5,6,7,8,9,10,11,12]).reshape(-1,1)
scores=np.array([35,44,56,58,69,65,65,74,85,85,94,96])

#split into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(hours,scores,test_size=0.2,random_state=43)

#train the model
model=LinearRegression()

model.fit(x_train,y_train)

#make prediction
predicted_y=model.predict(x_test)
print(predicted_y)

#evaluate the model
print("gradient (m): ",model.coef_[0])
print("intercept (c): ",model.intercept_)
print("Mean squared error: ",mean_squared_error(y_test,predicted_y))
print(" R2 score",r2_score(y_test,predicted_y))
plt.scatter(hours,scores,color="blue",label="Actual Data")
plt.plot(hours,model.predict(hours),label="regression Line")
plt.xlabel("Hours studies")
plt.ylabel("Marks Scored")
plt.title("Relation between sudy duration and performance of students ")
plt.legend()
plt.savefig("output_plot.png")

plt.show()
