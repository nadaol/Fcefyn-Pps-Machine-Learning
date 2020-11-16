import re
import matplotlib.pyplot as plot

# PATHS
file_name = "log.txt"
log_path = "./logs/"
file_path = log_path + file_name

# Abrir y leer log.txt

numEpoch = []       # lista de epochs
train_Loss = []        # train_loss
eval_Loss = []

f = open(file_path, "r")
linea = f.readlines()

indice = 0
for renglon in linea:

    Eval_line = re.search(r"Evaluation Set loss : (\d+.\d+)",renglon)
    if Eval_line is not None:
        eval_Loss.append(float(Eval_line.group(1)))

    Train_line = re.search(r"Epoch\s(\d+)\sLoss\s(\d+.\d+)", renglon)
    if Train_line is not None:
        numEpoch.append( int(Train_line.group(1)) )
        train_Loss.append( float(Train_line.group(2)) )

for i in range(len(numEpoch)):
    print("numEpoch: %d -- Train Loss: %f Evalutaion Loss : %f \n" % (numEpoch[i], train_Loss[i],eval_Loss[i]))

## Generar grafico y mostrar

plot.plot(numEpoch, train_Loss,'g',label='Training Loss')
plot.plot(numEpoch,eval_Loss,'r',label='Evalutaion Loss')
plot.legend(loc='lower left')
plot.xlabel("Epoch")
plot.ylabel("Losses")
plot.title("Losses in encoder text training")

plot.show()



