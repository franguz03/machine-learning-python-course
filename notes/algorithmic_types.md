lineal: its behavior is described by a lineal function
no linea: its behavior cannot be described by a lineal function
ensambled: they mix severeal predictions to get a better final prediction


instance based: this kind of algorithmics dont build explicit models, they store examples and make predictions based on the similarity with previous instances

## linear algorithmics

# lineal regression

this model draws a straight line that show us the tendency in a continuous dataset, it has very few parameters to configure



```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

modelo = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(modelo, X, y, cv=kf, scoring='r2')

print(f"Scores en validación cruzada: {scores}")
print(f"Promedio del score: {scores.mean()}")

```
# ridge regression RiR

it is an extension of LiR but the loss function es modificates to minimize the model complexity using a kind of internal feature selection procress (l2)

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold

modelo = Ridge(alpha=1.0)  # Puedes ajustar alpha según necesites
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(modelo, X, y, cv=kf, scoring='r2')

print(f"Scores en validación cruzada: {scores}")
print(f"Promedio del score: {scores.mean()}")
```
# lasso 
also an extension of LiR  as ridge but this one uses l1 norme

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold

modelo = Lasso(alpha=1.0)  # Puedes ajustar alpha según necesites
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(modelo, X, y, cv=kf, scoring='r2')

print(f"Scores en validación cruzada: {scores}")
print(f"Promedio del score: {scores.mean()}")
```

# elasticNet
another extension of LiR, but this mixes the RiR and lasso properties
```python
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, KFold

modelo = ElasticNet(alpha=1.0, l1_ratio=0.5)  # Ajusta alpha y l1_ratio según necesites
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(modelo, X, y, cv=kf, scoring='r2')

print(f"Scores en validación cruzada: {scores}")
print(f"Promedio del score: {scores.mean()}")
```

# logistic regression 

for binary calssification problems, it uses a sigmoid function that has an output range between  0 and 1
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold

modelo = LogisticRegression(max_iter=1000)  # Asegura suficientes iteraciones para convergencia
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(modelo, X, y, cv=kf, scoring='accuracy')  # Usamos precisión como métrica

print(f"Scores en validación cruzada: {scores}")
print(f"Promedio del score: {scores.mean()}")
```

# linear discriminant analysis
for classification problems, it uses a gaussian distribution por numeric inputs 
```python
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, KFold

modelo = LinearDiscriminantAnalysis()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(modelo, X, y, cv=kf, scoring='accuracy')  # Usamos precisión como métrica

print(f"Scores en validación cruzada: {scores}")
print(f"Promedio del score: {scores.mean()}")
```

## no linear algorithmics

# knn

instance bases algorithmic, it compares with n (odd)neighbors to decide the class

```python
from sklearn.neighbors import KNeighborsClassifier

# Crear el modelo KNN con k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Entrenar el modelo
knn.fit(X_train, y_train)

# Hacer predicciones
y_pred = knn.predict(X_test)




```
we can use kNeighborsClassifer or KNeighborsRegressor

# Naive Bayes
it makes the classifaction assuming the dependency between features, calculating the class according to bayesian probabilities

```python
from sklearn.naive_bayes import GaussianNB

# Crear el modelo Naive Bayes
nb = GaussianNB()

# Entrenar el modelo
nb.fit(X_train, y_train)

# Hacer predicciones
y_pred = nb.predict(X_test)
```

# Support vector machine
it creates hiperplanes do decide or class or the regression, n planes for n features
classification: SVC
Regressor: SVR

```python
from sklearn.svm import SVC

# Crear el modelo SVM
svm = SVC(kernel='rbf') # linear, poly , sigmoid, precomputed

# Entrenar el modelo
svm.fit(X_train, y_train)

# Hacer predicciones
y_pred = svm.predict(X_test)
```

# calssification and regression trees

it works bettere with descretal featues (dummy, categorical)

classfication: DecisionTreeClassifier
regression: DecisionTreeRegressor

## ensemble algorithmics

# Bagging

It builds several tree models: Random Forest, Extra Trees, and Bagged Decision Trees. In Voting, you choose how to combine the predictions. Usually, it averages the best ones.

```py
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Modelo base: Árbol de decisión
base_tree = DecisionTreeClassifier()

# Bagging con árboles de decisión
bagging = BaggingClassifier(base_estimator=base_tree, n_estimators=100, random_state=42)

# Entrenar el modelo
bagging.fit(X_train, y_train)

# Predicciones
y_pred = bagging.predict(X_test)


```

# boosting

It builds several models (usually of the same type), where each one learns to correct the errors of the previous model in the chain. Examples include AdaBoost and Stochastic Gradient Boosting.

```py
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Modelo base: Árbol de decisión
base_tree = DecisionTreeClassifier(max_depth=1)

# AdaBoost
adaboost = AdaBoostClassifier(base_estimator=base_tree, n_estimators=100, random_state=42)

# Gradient Boosting
gradient_boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Entrenar los modelos
adaboost.fit(X_train, y_train)
gradient_boosting.fit(X_train, y_train)

# Predicciones
y_pred_adaboost = adaboost.predict(X_test)
y_pred_gb = gradient_boosting.predict(X_test)
```
# Voting

It builds several models (usually of different types) and a supervised model that learns how to better combine the predictions of the primary models.

```py
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Modelos base
log_reg = LogisticRegression()
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
gradient_boosting = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Voting Classifier (combinación de modelos)
voting = VotingClassifier(estimators=[
    ('lr', log_reg),
    ('rf', random_forest),
    ('gb', gradient_boosting)
], voting='hard')  # 'hard' para votación por mayoría, 'soft' para promedio de probabilidades

# Entrenar el modelo
voting.fit(X_train, y_train)

# Predicción
y_pred = voting.predict(X_test)

```
It is not very commonly used, but it can be helpful when we have tested several models and all of them perform well. In this case, we can combine them to improve overall performance.