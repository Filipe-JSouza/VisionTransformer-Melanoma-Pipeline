
# VITs e Objetivo do Trabalho

Este projeto explora **Vision Transformers (ViTs)**, uma arquitetura baseada em mecanismos de atenção originalmente desenvolvida para processamento de linguagem natural, mas adaptada para visão computacional. Os ViTs dividem imagens em patches e aplicam atenção para capturar relações globais, oferecendo resultados competitivos em tarefas de classificação de imagens.

## Objetivo

O trabalho busca abordar os seguintes tópicos:

1. **Refinamento de um modelo pré-treinado** utilizando o banco de dados **[IMAGE OF MELANOMAS AND NAEVUS](https://www.cs.rug.nl/~imaging/databases/melanoma_naevi/index.html)**.
2. Integração com um classificador **LDA (Linear Discriminant Analysis)**.
3. Comparação de diferentes métodos de agregação de features, incluindo:
   - Média
   - Mediana
   - CLS Token
   - PCA (Principal Component Analysis)
   - Outros métodos relevantes
4. Avaliação da metodologia empregada utilizando os testes estatísticos **Friedman** e **Nemenyi**.
5. Determinação do melhor **método de agregação** para este contexto.

---

### Espaço reservado para imagem:
others database of melanomas:
- https://data.mendeley.com/datasets/zr7vgbcyr2/1

- https://derm.cs.sfu.ca/Download.html
