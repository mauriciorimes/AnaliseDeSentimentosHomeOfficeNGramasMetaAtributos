Usando a nova base dados extraída, depois de alguns tratamentos feitos anteriormente, esse algoritmo foi desenvolvido com as duas abordagens usadas antes, juntas, a fim de comparação. Usando Python com a biblioteca pandas.

Foram realizadas 10 avaliações com variações em sua precisão a fim de uso comparativo e um relatório final com a junção de todas.

Conclusão:
Com a base de dados tendo 4.32 vezes mais tweets neutros do que negativos e 3 vezes mais tweets positivos do que negativos, o algoritmo identifica de forma mais assertiva tweets positivos do que negativos. Separando os tweets neutros, mesmo com abordagens de alinhamento com a proporção de positivos para negativos. Olhando para o f1 score, podemos ver que a conclusão se mantem parecida com a precisão.

A melhor avaliação, foi:

                precisão    f1-score   

    NEGATIVO      0.421       0.302       
      NEUTRO      0.767       0.780       
    POSITIVO      0.655       0.685     

---> RESULTADOS DA VALIDAÇÃO CRUZADA COM 10 AVALIAÇÕES:
Acurácia: 0.6832
F1-score (macro): 0.5526  

Todas as avaliações e demais testes em: https://colab.research.google.com/drive/12sveEoBckcK-2yMv0zyhuRVKViBiIrAE?usp=sharing


