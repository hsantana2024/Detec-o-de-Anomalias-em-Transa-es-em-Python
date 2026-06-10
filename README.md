Projeto: Detecção de Anomalias em Transações Financeiras
Objetivo

Desenvolver um sistema capaz de identificar transações suspeitas ou fraudulentas analisando valores, horários e padrões de comportamento dos clientes.

Tecnologias Utilizadas
Python 3
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
Jupyter Notebook

Instalação:

pip install pandas numpy matplotlib seaborn scikit-learn
Estrutura do Projeto
deteccao_anomalias/
│
├── dados/
│   └── transacoes.csv
│
├── src/
│   ├── carregar_dados.py
│   ├── treinar_modelo.py
│   └── detectar_anomalias.py
│
├── resultados/
│   └── grafico_anomalias.png
│
└── main.py
Base de Dados

Arquivo: transacoes.csv

id,valor,hora
1,120,8
2,200,10
3,150,12
4,180,14
5,210,16
6,300,18
7,5000,3
8,180,11
9,220,15
10,7000,2

Os valores 5000 e 7000 representam possíveis anomalias.

Código Principal
main.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Carregar dados
dados = pd.read_csv("dados/transacoes.csv")

# Selecionar atributos
X = dados[["valor", "hora"]]

# Modelo de detecção de anomalias
modelo = IsolationForest(
    contamination=0.2,
    random_state=42
)

modelo.fit(X)

# Previsão
dados["anomalia"] = modelo.predict(X)

# Converter:
# -1 = anomalia
#  1 = normal

print(dados)

# Visualização
cores = dados["anomalia"].map({
    1: "blue",
    -1: "red"
})

plt.figure(figsize=(8,5))
plt.scatter(
    dados["hora"],
    dados["valor"],
    c=cores
)

plt.xlabel("Hora")
plt.ylabel("Valor")
plt.title("Detecção de Anomalias em Transações")
plt.grid()

plt.show()


Saída Esperada
   id  valor  hora  anomalia
0   1    120     8         1
1   2    200    10         1
2   3    150    12         1
3   4    180    14         1
4   5    210    16         1
5   6    300    18         1
6   7   5000     3        -1
7   8    180    11         1
8   9    220    15         1
9  10   7000     2        -1


Melhorias Futuras
Nível Intermediário
Analisar localização geográfica.
Identificar horários incomuns.
Detectar múltiplas transações em poucos segundos.
Gerar relatório em PDF.
Nível Avançado
Criar API com FastAPI.
Dashboard com Streamlit.
Banco de dados com PostgreSQL.
Alertas por e-mail.
Modelo em tempo real utilizando streaming de dados.
O que você aprende neste projeto
Manipulação de dados com Python.
Análise exploratória de dados.
Machine Learning não supervisionado.
Uso do algoritmo Isolation Forest.
Visualização de dados.
Construção de projetos reais para portfólio.

Esse projeto é muito valorizado em vagas de Data Analyst, Data Scientist, Machine Learning Engineer e Cybersecurity, pois simula um cenário real de detecção de fraudes bancárias.
