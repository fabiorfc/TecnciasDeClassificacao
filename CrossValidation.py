# -*- coding: utf-8 -*-
"""----------------------------------
    VALIDAÇÃO CRUZADA USANDO VALIDAÇÃO CRUZADA COM ESTRATIFICAÇÃO AMOSTRAL
"""


"""----------------------------------
    IMPORTANDO AS LIBRARIES
"""
from sklearn.model_selection import cross_validate,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


"""----------------------------------
    IMPORTAÇÃO E TRATAMENTO DOS DADOS
"""
#Carregando os dados
dados = pd.read_csv("...")
#Separando as features das labels
x = dados[["x1","x2","x3"]]
y = dados["y"]


"""----------------------------------
    PARAMETRIZAÇÃO
"""
modelo = DecisionTreeClassifier(max_depth=2)
pastas = 10



"""----------------------------------
    IMPLEMENTAÇÃO DO CROSS VALIDATION
"""
def CrossValidation(modelo, x, y, pastas):
    
    #Implementação
    cross_validation_cv = StratifiedKFold(n_splits = pastas, shuffle = True)
    resultados = cross_validate(modelo, x, y, cv = cross_validation_cv)
    #Validação dos resultados
    resultados_do_teste = resultados['test_score']
    
    #Cálculo do intervalor de confiança do score do teste
    media = resultados_do_teste.mean()
    desvio_padrao = resultados_do_teste.std()
    intervalo_inferior = round(media - 2*desvio_padrao, 4)
    intervalo_superior = round(media + 2*desvio_padrao, 4)

    #Resultados
    resultado_final = {"media":media, "intervalo_inferior":intervalo_inferior,"intervalo_superior":intervalo_superior}
    
    return resultado_final


CrossValidation(modelo, x, y, pastas)



