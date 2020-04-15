import numpy as np

class Perceptron(object):

    def __init__(ident, urutinput, nilaithresh=100, ratapembelajaran=0.01):
        ident.nilaithresh = nilaithresh
        ident.ratapembelajaran = ratapembelajaran
        ident.berat = np.zeros(urutinput + 1)
           
    def predict(ident, inputan):
        hasilj = np.dot(inputan, ident.berat[1:]) + ident.berat[0]
        if hasilj > 0:
          aktivasi = 1
        else:
          aktivasi = 0            
        return aktivasi

    def train(ident, inputdt, output):
        for _ in range(ident.nilaithresh):
            for inputan, label in zip(inputdt, output):
                prediksi = ident.predict(inputan)
                ident.berat[1:] += ident.ratapembelajaran * (label - prediksi) * inputan
                ident.berat[0] += ident.ratapembelajaran * (label - prediksi)


inputdt = []
inputdt.append(np.array([1, 1]))
inputdt.append(np.array([1, 0]))
inputdt.append(np.array([0, 1]))
inputdt.append(np.array([0, 0]))

output = np.array([1, 0, 0, 0])

perceptron = Perceptron(2)
perceptron.train(inputdt,output)

inputan = np.array([1, 1])
perceptron.predict(inputan) 
#=> 1

inputan = np.array([0, 1])
perceptron.predict(inputan) 
#=> 0