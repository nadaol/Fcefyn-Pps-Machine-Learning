import re
import matplotlib.pyplot as plot

# Plots evaluation & training losses of a model from a training log

# PATHS
file_name = "log_100epoch_eval.txt"
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
    else:
        Train_line = re.search(r"Epoch\s(\d+)\sLoss\s(\d+.\d+)", renglon)
        if Train_line is not None:
            numEpoch.append( int(Train_line.group(1)) )
            train_Loss.append( float(Train_line.group(2)) )

min_eval_loss = 100.0
min_train_loss = 100.0

for i in range(len(numEpoch)):
    if( (len(eval_Loss) > 0) and ( eval_Loss[i] < min_eval_loss) ):
        min_eval_loss = eval_Loss[i]
        min_eval_epoch = i
    if(train_Loss[i] < min_train_loss):
        min_train_loss = train_Loss[i]
        min_train_epoch = i
#   print("numEpoch: %d -- Train Loss: %f Evalutaion Loss : %f \n" % (numEpoch[i], train_Loss[i],eval_Loss[i]))

if(len(eval_Loss) > 0):
    print("Minimum evaluation loss at epoch %d : %f \n" % (min_eval_epoch,min_eval_loss) )

print("Minimum training loss at epoch %d : %f\n" % (min_train_epoch,min_train_loss) )

## Generar grafico y mostrar

plot.plot(numEpoch, train_Loss,'go--',label='Training Loss')
if(len(eval_Loss)>0):
    plot.plot(numEpoch,eval_Loss,'ro--',label='Evalutaion Loss')
plot.legend(loc='lower left')
plot.xlabel("Epoch")
plot.ylabel("Losses")
plot.title("Losses in encoder text training")

plot.show()
