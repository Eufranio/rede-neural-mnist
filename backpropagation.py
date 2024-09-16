import numpy as np
import random

class RedeNeural(object):

    # quantidade_camadas: quantidade de camadas ocultas. neuronios_por_camada: quantidade
    # de neuronios que cada camada, de 1 até quantidade_camadas, terá.
    # ex: quantidade_camadas=3, neuronios_por_camada=[5,4,3]
    def __init__(self, quantidade_camadas, neuronios_por_camada):
        self.quantidade_camadas = quantidade_camadas
        self.neuronios_por_camada = neuronios_por_camada
        self.classes = neuronios_por_camada[-1]

        self.pesos = []
        # iniciando vieses com zero
        self.vieses = [np.zeros((qtd_neuronios, 1)) for qtd_neuronios in neuronios_por_camada[1:]]

        # não pegamos a primeira camada pois não ha neuronios da camada anterior para
        # a primeira camada = camada de entrada
        for (i, j) in zip(neuronios_por_camada[:-1], neuronios_por_camada[1:]):
            # i: quantidade de neuronios na camada de origem
            # j: quantidade de neuronios na camada de destino
            # iniciando com pesos aleatorios
            self.pesos.append(np.random.normal(loc=0, scale=np.sqrt(2/i), size=(j, i)))

    # mini_batch é um mini batch de vários (x, y)
    def calcular_por_mini_batch(self, mini_batch, taxa_aprendizado):
        # lembrar de zerar os gradientes
        gradientes_vies = [np.zeros(v.shape) for v in self.vieses]
        gradientes_peso = [np.zeros(p.shape) for p in self.pesos]

        for x, y in mini_batch:
            correcao_vies, correcao_peso = self.backpropagation(x, y)
            # atualização dos gradientes
            for i in range(len(correcao_vies)):
                gradientes_vies[i] += correcao_vies[i]
                gradientes_peso[i] += correcao_peso[i]

        # atualização dos pesos
        # w = w - (taxa / N) * erro
        for i in range(len(self.pesos)):
            self.pesos[i] = self.pesos[i] - (taxa_aprendizado / len(mini_batch)) * gradientes_peso[i]
        for i in range(len(self.vieses)):
            self.vieses[i] = self.vieses[i] - (taxa_aprendizado / len(mini_batch)) * gradientes_vies[i]

    # usado para calcular os gradientes
    def backpropagation(self, x, y):
        gradientes_vies = [np.zeros(v.shape) for v in self.vieses]
        gradientes_peso = [np.zeros(p.shape) for p in self.pesos]

        ativacao = x
        ativacoes = [x]
        zs = []

        for i in range(len(self.pesos) - 1):
            # h = ativacao(z) = ativacao(b+wi.xi)
            # z = vies + (pesos que conectam a camada anterior) * (ativações da camada anterior)
            z = self.vieses[i] + np.dot(self.pesos[i], ativacao)
            zs.append(z)
            ativacao = sigmoide(z)
            ativacoes.append(ativacao)

        # adapta a ultima camada para softmax
        z = self.vieses[-1] + np.dot(self.pesos[-1], ativacao)
        zs.append(z)
        activation = softmax(z)
        ativacoes.append(activation)

        # calcular o delta
        delta = derivada_softmax(ativacoes[-1], y)
        gradientes_vies[-1] = delta
        gradientes_peso[-1] = np.dot(delta, ativacoes[-2].transpose())

        # propagar o gradiente de trás para frente
        for l in range(2, self.quantidade_camadas):
            z = zs[-l]
            derivada = derivada_sigmoide(z)
            delta = np.dot(self.pesos[-l+1].transpose(), delta) * derivada
            gradientes_vies[-l] = delta
            gradientes_peso[-l] = np.dot(delta, ativacoes[-l-1].transpose())
        return gradientes_vies, gradientes_peso

    def calcular(self, test_data):
        perda_total = 0.0
        acertos = 0

        # extraindo enradas e saidas para transformar em one-hot
        inputs = [x for x, _ in test_data]
        labels = [y for _, y in test_data]

        # rótulos para o formato one-hot
        one_hot = np.eye(self.classes)[labels]

        # uma lista de tuplas (entrada, rótulo one-hot)
        test_data_one_hot = zip(inputs, one_hot)

        # y deve ser o vetor one-hot para podermos usar a função de custo de entropia cruzada
        for x, y in test_data_one_hot:
            ativacoes = self.feedforward(x)
            erro = entropia_cruzada(ativacoes, y)
            perda_total += erro

            saida_prevista = np.argmax(ativacoes)
            saida_real = np.argmax(y)
            if saida_prevista == saida_real:
                acertos += 1

        total = len(test_data)
        # acertos, custo, acuracia
        return acertos, perda_total/total, acertos/total

    def feedforward(self, a):
        for i in range(len(self.vieses) - 1):
            z = np.dot(self.pesos[i], a) + self.vieses[i]
            a = sigmoide(z)

        z = self.vieses[-1] + np.dot(self.pesos[-1], a)
        a = softmax(z)
        return a

    def treinar(self, training_data, epochs, mini_batch_size, eta, test_data):
        training_data = list(training_data)
        n = len(training_data)

        test_data = list(test_data)
        n_test = len(test_data)

        custos_por_epoca = []
        acuracias_por_epoca = []

        for j in range(1, epochs+1):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.calcular_por_mini_batch(mini_batch, eta)

            acertos, custo, acuracia = self.calcular(test_data)
            print("Epoca {}: {} / {}, custo={}, acuracia={}".format(
                j,
                acertos,
                n_test,
                custo,
                acuracia))

            custos_por_epoca.append(custo)
            acuracias_por_epoca.append(acuracia)

        return custos_por_epoca, acuracias_por_epoca

def entropia_cruzada(saida_prevista, saida_real):
    return np.sum(saida_real * np.log(saida_prevista))

def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))

# Função para retornar as derivadas da função Sigmóide
def derivada_sigmoide(z):
    return sigmoide(z)*(1-sigmoide(z))

def softmax(z):
    z = z - np.max(z) # Estabiliza os valores para evitar overflow
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0)

# Função de derivada para softmax com entropia cruzada
def derivada_softmax(ativacoes, y):
    return ativacoes - y

