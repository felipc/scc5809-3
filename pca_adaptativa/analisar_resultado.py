
# coding: utf-8

# PCA Clássica para classificar banco de dados iris.dat

# ### Ler dados
# Primeiramente vamos importar as bibliotecas necessárias e ler o arquivos de dados iris.dat

import numpy as np

from pybrain.tools.shortcuts import *
from pybrain.datasets import *
from pybrain.supervised.trainers import *
from pybrain.structure.modules import *

from matplotlib.pyplot import *

np.set_printoptions(precision=4, suppress=True)


def ler_dados(filename):
    training_file = open(filename, 'r')

    input_array = []
    output_array = []

    for x in training_file.readlines():
        if (x.strip() == ''):
            continue

        line = x.strip().split(',')
        
        this_input = [float(num) for num in line[:-1]]
        this_output = int(line[-1])
        
        input_array.append(this_input)
        output_array.append(this_output)

    print "Added %d entries in the dataset" % len(input_array)

    assert(len(input_array) == len(output_array))

    return input_array, output_array


###### 
#
# Ler dados
#


input_array, output_array = ler_dados('saida.csv')



unique_info = np.unique(output_array, return_counts=True)
print "unique info: ", unique_info


# In[26]:

new_input = np.array(input_array)



###### 
#
# Exibindo resultados
#

figures = [
    "ro",  #red circle
    "bo",  #blue circle
    "gx"   #green X
    ]

figure(1)
for i in range(len(input_array)):
    point = new_input[i]
    plot(point[0], point[1], figures[output_array[i]])

show()


###### 
#
# MLP
#


def generate_dataset(input, target):
    DS = ClassificationDataSet(len(input[0]), class_labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    for i in range(0, len(input)):
        DS.addSample(input[i], target[i])
    return DS


DS = generate_dataset(new_input, output_array)

# Verificar que os dados estão como esperado
print "estatisticas do dataset: ", DS.calculateStatistics()




DS._convertToOneOfMany()



TrainDS, TestDS = DS.splitWithProportion(0.75)

print "Treino: %d, Teste: %d" % (len(TrainDS), len(TestDS))


# In[369]:

def criar_rede(dataset, learningrate=0.5):
    # Criando a rede neural
    network = buildNetwork(dataset.indim, 5, dataset.outdim, bias=True, outclass = SoftmaxLayer)
    
    # Criando o objeto "trainer" da rede neural
    trainer = BackpropTrainer(network, dataset, learningrate)
    return network, trainer

network, trainer = criar_rede(TrainDS)


# In[370]:

def treinar_rede(trainer, targetErr=0.01):
    # Treinando a rede
    treinos = 0
    err = 10000
    while err > targetErr:
        err = trainer.train()
        treinos += 1

        #Exibir progresso a cada 100 iterações
        if treinos % 20 == 0:
            print "Progresso: treinos: %d, erro: %f" % (treinos, err)

        if treinos > 1000:
            break

    print "\nTotal de treinos: %d" % treinos
    print "            Erro: %f\n" % err
    

treinar_rede(trainer)


# In[371]:

def validar_dados(network, dataset, verbose=False):
    errors = 0;
    input = dataset.getField('input')
    output = dataset.getField('target')
    
    for i in range(len(dataset)):
        netw = [int(round(x)) for x in network.activate(input[i])]
        expc = [x for x in output[i]]
    
        if (netw != expc):
            errors += 1

        if verbose:
            print netw, " == ", expc, "%s" % "" if netw == expc else " error!"

    print "\nErrors: %d out of %d (%f%%)" % (errors, len(dataset), float(errors) / len(dataset) * 100)

validar_dados(network, TestDS, verbose=True)




###### 
#
# MLP com dados originais
#


# OriginalDS = generate_dataset(normalized_input, output_array)
# OriginalDS._convertToOneOfMany()
# OriginalTrainDS, OriginalTestDS = OriginalDS.splitWithProportion(0.75)
# print "Original Treino: %d, Teste: %d" % (len(OriginalTrainDS), len(OriginalTestDS))
# network, trainer = criar_rede(OriginalTrainDS)
# treinar_rede(trainer)
# validar_dados(network, OriginalTestDS, verbose=True)
