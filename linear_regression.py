import pandas as pd
import matplotlib.pyplot as plt
# Veri okuma ve gorsellestirme
data = pd.read_csv("direnc.csv")
gerilim = data.Gerilim.values.reshape(-1,1)
akim = data.Akim.values.reshape(-1,1)

plt.scatter(gerilim,akim,label="Örnekler")
plt.xlabel("Gerilim[V]")
plt.ylabel("Akım[A]")
plt.legend()
# ------------------------------
# Model olusturma ve gorsellestirme
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=False)
model.fit(gerilim,akim)

tahminler = model.predict(gerilim)

plt.plot(gerilim,tahminler,"red",label="Linear Model")
plt.legend()
# ------------------------------
# Model denklemini ogrenme
print("2.5V gerilimindeki akım değeri: ",model.predict(2.5))
print("Denklem: ","V=",model.intercept_,"+",model.coef_,"*I")
# ------------------------------
