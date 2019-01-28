import pandas as pd
import matplotlib.pyplot as plt
# Veri okuma ve gorsellestirme
data = pd.read_csv("diyot.csv")
gerilim = data.Gerilim.values.reshape(-1,1)
akim = data.Akim.values.reshape(-1,1)

plt.scatter(gerilim,akim,label="Örnekler")
plt.xlabel("Gerilim[V]")
plt.ylabel("Akım[mA]")
plt.legend()
# ------------------------------
# Linear model olusturma ve gorsellestirme
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=False)
model.fit(gerilim,akim)
tahminler = model.predict(gerilim)

plt.plot(gerilim,tahminler,"red",label="Linear Model")
plt.legend()
# ------------------------------
# Polynomial model olusturma ve gorsellestirme
from sklearn.preprocessing import PolynomialFeatures
model2 = PolynomialFeatures(degree=4)
gerilim_poly = model2.fit_transform(gerilim) 

model3 = LinearRegression(fit_intercept=False)
model3.fit(gerilim_poly,akim)
tahminler2 = model3.predict(gerilim_poly)

plt.plot(gerilim,tahminler2,"black",label="Polynomial Model")
plt.legend()
# ------------------------------
