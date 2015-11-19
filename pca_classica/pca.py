
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
        
        output_name = line[-1]
        name_map = {
            "Iris-setosa": 0,
            "Iris-versicolor": 1,
            "Iris-virginica": 2
        }
        #Converte as strings do iris para valores 0, 1, 2
        this_output = name_map.get(output_name) #output value
        
        input_array.append(this_input)
        output_array.append(this_output)
        
        #print "Added entry: ", this_input, " => ", output_name, "(%d)" % this_output

    print "Added %d entries in the dataset" % len(input_array)

    assert(len(input_array) == len(output_array))

    return input_array, output_array


###### 
#
# Ler dados
#


input_array, output_array = ler_dados('iris.data')



unique_info = np.unique(output_array, return_counts=True)
print "unique info: ", unique_info


# In[26]:

input_array = np.array(input_array)


###### 
#
# Normalização
# 
# Normalização para distribuição Gaussiana, mean=0 and variance=1.

# Para fazer isso, subtraímos todos pela média (para atingir média = 0)
# e depois dividimos pelo desvio padrão (para atingiar variança = 1)
#
# Essa operação é feita coluna à coluna

# Primeiro transformamos nossa coluna de entrada numa entrada de array do numpy
normalized_input = np.array(input_array)
normalized_input = normalized_input - normalized_input.mean(axis=0)
normalized_input = normalized_input / normalized_input.std(axis=0)
assert(normalized_input.shape == (150, 4))


###### 
#
# Matriz de covariância
#

norm_cov = np.cov(normalized_input.T)

print "matriz de covariancia", norm_cov

###### 
#
# Auto-valores e auto-vetores
#


eig = np.linalg.eig(norm_cov)

eigvals = eig[0]

# The documentation of np.linalg.eig says:
# - The normalized (unit “length”) eigenvectors, such that the column
# - v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
#
# This means that the eigenvectors are transposed

eigvecs = eig[1].T



###### 
#
# Contribuicoes
#

#contribuicoes de cada dimensao
contribs = eigvals / eigvals.sum()
print "contribuicoes de cada dimensao: ", contribs

#
#  Atencao: por coincidencia, no iris.data os autovetores ja sairam
#  ordenados. Caso uma outra base de dados seja usada, é preciso ordená-los
#  por contribuição
#


print "Contribuicoes cumulativas"
#contribuicoes cumulativas
acumulado = 0
for i in range(len(contribs)):
    acumulado += contribs[i]
    print "%d dimensao: %.2f%%" % (i + 1, acumulado * 100)



###### 
#
# Reduzir dimensionalidade
#


#Dimensoes à serem utilizadas
vecs_to_use = 2

new_input = np.ndarray((len(input_array), vecs_to_use))
new_input.shape

for i in range(len(input_array)):
    for j in range(vecs_to_use):
        new_input[i][j] = normalized_input[i].dot(eigvecs[j])



#This is how it can be sorted
#eig_vals, eig_vecs = np.linalg.eig(norm_cov)
# Make a list of (eigenvalue, eigenvector) tuples
#eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
#eig_pairs.sort()
#eig_pairs.reverse()
#matrix_w = np.hstack((eigvecs[0].reshape(4,1),
#                      eigvecs[1].reshape(4,1)))
#Y = normalized_input.dot(matrix_w)
# Y é o mesmo que new_input



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

        if treinos > 2000:
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


OriginalDS = generate_dataset(normalized_input, output_array)
OriginalDS._convertToOneOfMany()
OriginalTrainDS, OriginalTestDS = OriginalDS.splitWithProportion(0.75)
print "Original Treino: %d, Teste: %d" % (len(OriginalTrainDS), len(OriginalTestDS))
network, trainer = criar_rede(OriginalTrainDS)
treinar_rede(trainer)
validar_dados(network, OriginalTestDS, verbose=True)
