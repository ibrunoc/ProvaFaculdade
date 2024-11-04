import pandas as pd
import matplotlib.pyplot as plt

def calcular_media(lista):
    media = sum(lista) / len(lista)
    print("A Media das idades é igual a", media)

idades = [24, 23, 45, 33, 24, 33, 42, 38, 33, 45, 24, 33]
calcular_media(idades)

def distribuicao_frequencia(lista):
    df = pd.DataFrame(lista, columns=['Idades'])
    distribuicao = df['Idades'].value_counts().sort_index()
    print(distribuicao)
    return distribuicao

distribuicao = distribuicao_frequencia(idades)


def grafico_distribuicao(distribuicao):
    distribuicao.plot(kind='bar', color='skyblue')
    plt.xlabel("Idades")
    plt.ylabel("Frequência")
    plt.title("Distribuição de Frequência das Idades")
    plt.show()

grafico_distribuicao(distribuicao)
