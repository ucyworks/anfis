import anfis
import membership.mfDerivs
import membership.membershipfunction
import numpy
import pandas as pd

# Synthethic-data.csv dosyasından veri okuma
data = pd.read_csv("../../synthethic-data.csv")

# İlk %80'i eğitim verisi, son %20'si test verisi
train_size = int(len(data) * 0.8)
train_data = data[:train_size]

# X: giriş değişkenleri (sic, graphite, weight, sliding_rate)
# Y: çıkış değişkeni (wear_rate)
X = train_data[['sic', 'graphite', 'weight', 'sliding_rate']].values
Y = train_data['wear_rate'].values

mf = [[['gaussmf',{'mean':0.,'sigma':1.}],['gaussmf',{'mean':-1.,'sigma':2.}],['gaussmf',{'mean':-4.,'sigma':10.}],['gaussmf',{'mean':-7.,'sigma':7.}]],
      [['gaussmf',{'mean':1.,'sigma':2.}],['gaussmf',{'mean':2.,'sigma':3.}],['gaussmf',{'mean':-2.,'sigma':10.}],['gaussmf',{'mean':-10.5,'sigma':5.}]],
      [['gaussmf',{'mean':15.,'sigma':5.}],['gaussmf',{'mean':25.,'sigma':5.}],['gaussmf',{'mean':35.,'sigma':10.}]],
      [['gaussmf',{'mean':0.5,'sigma':0.5}],['gaussmf',{'mean':1.5,'sigma':0.5}],['gaussmf',{'mean':2.5,'sigma':0.5}]]]


mfc = membership.membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(X, Y, mfc)
anf.trainHybridJangOffLine(epochs=20)
print(round(anf.consequents[-1][0],6))
print(round(anf.consequents[-2][0],6))
print(round(anf.fittedValues[9][0],6))
if round(anf.consequents[-1][0],6) == -5.275538 and round(anf.consequents[-2][0],6) == -1.990703 and round(anf.fittedValues[9][0],6) == 0.002249:
	print('test is good')

print("Plotting errors")
anf.plotErrors()
print("Plotting results")
anf.plotResults()
