import re
import matplotlib.pyplot as plot

# PATHS
file_name = "log.txt"
log_path = "/home/admint/PycharmProjects/GraficoLogLoss/logs/"
file_path = log_path + file_name

# Abrir y leer log.txt

totalEpochs = 60    # cantidad total de epochs
numEpoch = []       # lista de epochs
valLoss = []        # lista de loss

f = open(file_path, "r")
linea = f.readlines()

indice = 0
for renglon in linea:
    #print("Linea %d: %s" %(indice,renglon))
    resultado = re.search(r"Epoch\s(\d+)\sLoss\s(\d+.\d+)", renglon)
    if resultado is not None:
        numEpoch.append( int(resultado.group(1)) )
        valLoss.append( float(resultado.group(2)) )

    ## Imprimir lista dual

for i in range(len(numEpoch)):
    print("numEpoch: %d -- Loss: %f" % (numEpoch[i], valLoss[i]))

## Generar grafico y mostrar

plot.plot(numEpoch, valLoss)
plot.xlabel("Epoch")
plot.ylabel("Loss")
plot.title("CURVA DE LOSS")

plot.show()



