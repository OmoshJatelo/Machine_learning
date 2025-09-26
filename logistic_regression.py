import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


np.random.seed(42)
#np.random.normal(mean,std,size)
previous_score=np.random.normal(60,6,2000)
study_hours=np.random.normal(5,2,2000)
attendance=np.random.normal(75,5,2000)

pass_probability=1/(1+np.exp(-(0.41*study_hours+0.05*attendance+0.06*previous_score-9)))
has_passed=np.random.binomial(1,pass_probability)
#print(has_passed)

data=pd.DataFrame({
    "Previous_score":previous_score,
    "Hours_studied":study_hours,
    "Attendance":attendance,
    "Pass_probability":pass_probability,
    "Passed":has_passed
})
print(data.head())

x_data=data[['Previous_score','Hours_studied','Attendance']]
y_data=data['Passed']

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print("Acurracy: \n",accuracy_score(y_test,y_predict))
print("Confusion Matrix: \n",confusion_matrix(y_test,y_predict))
print("Classification Report: \n",classification_report(y_test,y_predict))

sns.heatmap(confusion_matrix(y_test,y_predict),annot=True,fmt="d",cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("Confusion_matrix.png")
plt.show()

plt.scatter(study_hours,previous_score)
plt.savefig("result_plot.png")
plt.show()
