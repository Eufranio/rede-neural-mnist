# Implementação de Rede Neural para o Dataset MNIST

Este repositório contém uma implementação de uma rede neural feedforward utilizando **NumPy** para classificar dígitos manuscritos do dataset **MNIST**. O projeto foi desenvolvido como parte da disciplina de **Aprendizado de Máquina**.

## Descrição

A rede neural implementada consiste em:
- Uma camada de entrada de 784 neurônios (correspondendo aos pixels de uma imagem de 28x28),
- Camadas ocultas configuráveis,
- Uma camada de saída de 10 neurônios (correspondendo às classes de dígitos de 0 a 9).

A função de ativação usada nas camadas ocultas é a **sigmoide**, e a última camada usa **softmax** para calcular a distribuição de probabilidade sobre as classes. Para o cálculo do erro, utilizamos a função de custo **entropia cruzada**.

O treinamento é realizado por meio de **Stochastic Gradient Descent** (SGD) com **backpropagation**, e os dados são divididos em mini-batches.

## Funcionalidades

- Treinamento da rede neural com diferentes configurações de camadas ocultas e neurônios.
- Avaliação da rede nos dados de teste após cada época de treinamento.
- Uso de **softmax** na última camada para classificação.
- Função de custo baseada em **entropia cruzada**.
- Função de ativação **sigmoide** nas camadas ocultas.
- Otimização usando **Stochastic Gradient Descent** (SGD).

## Requisitos

- Python 3.x
- NumPy
- Matplotlib (para visualização, opcional)
- MNIST dataset (pode ser carregado via `mnist_loader`)

## Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/eufranio/rede-neural-mnist.git
   cd rede-neural-mnist
   ```

2. Instale as dependências:
    ```bash
    pip install numpy matplotlib
   ```
   
3. Execute o código para treinar e avaliar a rede neural:
    ```bash
   python graficos.py
   ```

## Parâmetros de Treinamento

Você pode ajustar os parâmetros de treinamento, como o número de épocas, tamanho dos mini-batches, e a taxa de aprendizado diretamente no arquivo `graficos.py`. Exemplo:

```python
net.treinar(training_data, epochs=5, mini_batch_size=10, eta=3.0, test_data=test_data)
```

## Dataset

O dataset utilizado é o MNIST, que contém imagens de dígitos manuscritos (de 0 a 9) em uma resolução de 28x28 pixels, com um total de 60.000 imagens de treino e 10.000 imagens de teste.
   

   