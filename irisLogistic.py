
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()

features = iris.data
labels = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=.3)

from sklearn.linear_model import LogisticRegression
lor = LogisticRegression()
lor.fit(X_train, Y_train)    
p= lor.predict(X_test)
print ("Accuracy2 = ",accuracy_score(Y_test,p))









