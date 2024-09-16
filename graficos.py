import random

import matplotlib.pyplot as plt
import mnist
import numpy as np

from backpropagation import RedeNeural

def run(epocas,acuracia_xmin=0,custo_xmin=0):
    custos, acuracia = run_network(epocas)
    make_plots(custos, acuracia, epocas, acuracia_xmin, custo_xmin)

def run_network(epocas):
    # Torne os resultados mais facilmente reproduzíveis
    random.seed(12345678)
    np.random.seed(12345678)

    treinamento, teste = mnist.load()

    net = RedeNeural(3, [28*28, 30, 10])
    custos_por_epoca, acuracias_por_epoca = (
        net.treinar(treinamento, epocas, 10, 0.5, test_data=teste))

    return custos_por_epoca, acuracias_por_epoca

def make_plots(custos, acuracia, num_epochs,
               test_accuracy_xmin=200,
               test_cost_xmin=0):
    plot_acuracias_por_epoca(acuracia, num_epochs, test_accuracy_xmin)
    plot_custos_por_epoca(custos, num_epochs, test_cost_xmin)

def plot_acuracias_por_epoca(acuracia, num_epochs, xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        np.arange(xmin, num_epochs),
        [acuracia for acuracia in acuracia[xmin:num_epochs]],
        color='#2A6EA6')
    ax.set_xlim([xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoca')
    ax.set_title('Acuracia para os testes')
    plt.show()

def plot_custos_por_epoca(custo, num_epochs, xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        np.arange(xmin, num_epochs),
        [erro for erro in custo[xmin:num_epochs]],
        color='#2A6EA6')
    ax.set_xlim([xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoca')
    ax.set_title('Custos para os testes')
    plt.show()

if __name__ == "__main__":
    epocas = int(input(
        "épocas: "))
    run(epocas)