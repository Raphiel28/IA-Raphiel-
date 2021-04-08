import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
regr = linear_model.LinearRegression()



datos = pd.read_csv("movies2.csv")
df = pd.DataFrame(datos)

x = df["movie_facebook_likes"]
y = df["imdb_score"]

Xx=x[:,np.newaxis]
regr.fit(Xx,y)
regr.coef_
m = regr.coef_[0]
b = regr.intercept_
y_p = m*Xx+b
print("prediccion", y_p)
print('y={0}*x+{1}'.format(m,b))
print(regr.predict(Xx)[0:5])
print(r2_score(y,y_p))

plt.scatter(x,y,color="blue")
plt.plot(x,y_p,color="red")
plt.title("Regresion Lineal", fontsize=16)
plt.xlabel("Like Facebook",fontsize=13)
plt.ylabel("Calificacion_IMDB",fontsize=13)
plt.show()