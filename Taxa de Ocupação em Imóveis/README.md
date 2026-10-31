# 🏘️ Quais fatores mais influenciam a taxa de ocupação de imóveis?

> Modelagem estatística com **Regressão Linear (OLS)** para entender quais características de um bairro explicam a **taxa de ocupação das residências**.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-OLS-4B8BBE)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

---

## 📑 Sumário

- [🎯 Objetivo](#-objetivo)
- [🗂️ Dados](#️-dados)
- [🧭 Fases do projeto](#-fases-do-projeto)
- [🧠 Decisões tomadas](#-decisões-tomadas)
- [📈 Evolução dos modelos](#-evolução-dos-modelos)
- [🏆 Resultados finais](#-resultados-finais)
- [⚠️ Limitações](#️-limitações)
- [🚀 Próximos passos](#-próximos-passos)
- [🛠️ Como executar](#️-como-executar)

---

## 🎯 Objetivo

Descobrir **quais fatores mais influenciam a taxa de ocupação das residências** nos bairros de uma cidade.

> 💡 **Importante:** este projeto é de **modelagem estatística**, não de Machine Learning preditivo. O foco é estudar a **relação entre as variáveis** e interpretar os coeficientes. Prever valores com a maior precisão possível não é o objetivo principal.

---

## 🗂️ Dados

- 📄 Arquivo: `dataset.csv`
- 📏 **500 observações** (bairros) e **14 colunas**, todas numéricas
- ✅ **Nenhum valor ausente**
- 🎯 **Variável alvo:** `taxa_ocupacao`

| Variável | Tipo |
|---|---|
| `numero_medio_quartos_por_residencia` | contínua |
| `idade_media_residencias` | contínua |
| `taxa_criminalidade` | contínua |
| `proporcao_lotes` | contínua |
| `proporcao_empresas` | contínua |
| `rio_no_bairro` | binária (0/1) |
| `taxa_poluicao` | contínua |
| `distancia_centro` | contínua |
| `index_acessibilidade` | discreta |
| `imposto_residencial` | discreta |
| `taxa_professores` | contínua |
| `consumo_medio_energia` | contínua |
| `taxa_desabrigados` | contínua |
| **`taxa_ocupacao`** 🎯 | **alvo** |

---

## 🧭 Fases do projeto

### 1️⃣ 🔍 Análise Exploratória (univariada)
- Verificação de formato, tipos de dados e valores nulos.
- Resumo estatístico com `describe()`.
- Identificação de possíveis **outliers** em `taxa_criminalidade` (de 0,006 até 88,97) e `proporcao_lotes`.
- Análise de **assimetria (skewness)** de todas as variáveis.
- A variável alvo estava distorcida (assimetria de **1,10**), então foi aplicada uma **transformação logarítmica**. A assimetria caiu para **-0,33**. ✨

### 2️⃣ 🔗 Análise Bivariada
- **Mapa de calor** de correlação entre todas as variáveis.
- Gráficos de dispersão (`regplot`) e coeficiente de Pearson para os pares mais correlacionados.
- Objetivo: detectar **multicolinearidade** entre as preditoras e entender a relação delas com a variável alvo.

| Par de variáveis | Correlação |
|---|---|
| `proporcao_empresas` × `taxa_poluicao` | **+0,76** |
| `idade_media_residencias` × `taxa_poluicao` | **+0,73** |
| `distancia_centro` × `taxa_poluicao` | **-0,77** |
| `proporcao_empresas` × `imposto_residencial` | **+0,72** |
| `idade_media_residencias` × `distancia_centro` | **-0,75** |
| `index_acessibilidade` × `imposto_residencial` | **+0,91** ⚠️ (cai para **+0,24** sem outliers) |
| `numero_medio_quartos_por_residencia` × `taxa_ocupacao` | **+0,70** |
| `taxa_desabrigados` × `taxa_ocupacao` | **-0,74** |

### 3️⃣ ✂️ Preparação dos dados
- Separação em **X** (variáveis de entrada) e **Y** (`taxa_ocupacao` com log).
- Divisão **treino/teste 80/20** (`random_state=1`) → **400 treino / 100 teste**.

### 4️⃣ 🧪 Modelagem iterativa (5 versões)
Cada versão do modelo corrige um problema da anterior. Veja a tabela de [evolução dos modelos](#-evolução-dos-modelos).

### 5️⃣ ✅ Validação das suposições da regressão linear
Testes aplicados ao modelo final (v5):

| # | Suposição | Teste | Resultado |
|---|---|---|---|
| 1 | Linearidade | Rainbow Test | ✅ p = 0,73 |
| 2 | Independência dos erros | Durbin-Watson | ✅ 2,01 |
| 3 | Homocedasticidade | Goldfeld-Quandt | ✅ Satisfeita |
| 4 | Normalidade dos resíduos | Shapiro-Wilk + QQ-Plot | ❌ p ≈ 1,3e-10 |
| 5 | Sem multicolinearidade | VIF | ✅ Todos < 5 |

### 6️⃣ 📊 Interpretação dos coeficientes
Extração dos coeficientes do modelo final e tradução deles em insights.

---

## 🧠 Decisões tomadas

| 🤔 Decisão | 💬 Por quê? |
|---|---|
| **Log na variável alvo** | Reduzir a assimetria causada por valores extremos e aproximar a variável de uma distribuição normal. |
| **Outliers não removidos do modelo** | A remoção (`imposto_residencial < 600`) foi usada **apenas em uma cópia** (`df1`) para provar que a correlação de 0,91 era causada por outliers. O modelo usou os dados completos. |
| **Padronização com `StandardScaler`** | O VIF inicial estava absurdamente alto (até ~89) por causa da escala. Após padronizar, o diagnóstico ficou confiável. |
| **`fit` do scaler só no treino** | Evitar **vazamento de dados** (*data leakage*). No teste, apenas `transform`. |
| **Remover variáveis com VIF > 5** | `index_acessibilidade` (8,2) e `imposto_residencial` (10,0) saíram por multicolinearidade. |
| **Remover variáveis com valor-p > 0,05** | `idade_media_residencias`, `proporcao_empresas` e `consumo_medio_energia` saíram, e depois `proporcao_lotes`. |
| **Aplicar as mesmas remoções no teste** | Toda alteração feita no treino precisa ser replicada no teste para manter a consistência. |
| **Aceitar o modelo mesmo com resíduos não-normais** | Os gráficos mostraram desvio pequeno, e com amostras suficientemente grandes o teste t e a regressão OLS são robustos à não-normalidade. O modelo foi considerado "bom o suficiente". |

---

## 📈 Evolução dos modelos

| Versão | O que mudou | Nº de variáveis | R² | R² ajustado |
|---|---|---|---|---|
| **v1** | Dados brutos (sem tratamento) | 13 | 0,802 | 0,796 |
| **v2** | Dados padronizados | 13 | 0,802 | 0,796 |
| **v3** | Sem multicolinearidade (VIF) | 11 | 0,792 | 0,786 |
| **v4** | Sem variáveis com p > 0,05 | 8 | 0,789 | 0,785 |
| **v5** ⭐ | Sem `proporcao_lotes` (p > 0,05) | **7** | **0,788** | **0,784** |

> 🎉 O R² praticamente **não caiu** (de 0,802 para 0,788), mas o modelo ficou **muito mais simples, interpretável e confiável**: 7 variáveis, todas significativas e sem multicolinearidade.
>
> ℹ️ A v2 tem o mesmo R² da v1 porque padronizar as variáveis não altera o ajuste, apenas a escala dos coeficientes. Ela serviu para diagnosticar o VIF corretamente.

---

## 🏆 Resultados finais

### 📌 Modelo final (v5): 7 variáveis significativas

Como os dados estão **padronizados** e a variável alvo está em **escala logarítmica**, cada coeficiente representa a mudança em `log(taxa_ocupacao)` para o aumento de **1 desvio padrão** na variável. A coluna "efeito aproximado" converte isso em variação percentual (`exp(coef) - 1`).

| Variável | Coeficiente | Efeito aproximado | Valor-p |
|---|---|---|---|
| 🛌 `numero_medio_quartos_por_residencia` | **+0,0829** | ≈ +8,6% | < 0,001 |
| 🌊 `rio_no_bairro` | **+0,0297** | ≈ +3,0% | 0,003 |
| 🚨 `taxa_criminalidade` | **-0,0827** | ≈ -7,9% | < 0,001 |
| 🏭 `taxa_poluicao` | **-0,0838** | ≈ -8,0% | < 0,001 |
| 📍 `distancia_centro` | **-0,0889** | ≈ -8,5% | < 0,001 |
| 👩‍🏫 `taxa_professores` | **-0,0715** | ≈ -6,9% | < 0,001 |
| 🏚️ `taxa_desabrigados` | **-0,2131** | ≈ -19,2% | < 0,001 |

### 💡 Principais insights

- 🥇 **`taxa_desabrigados` é o fator mais forte**: bairros com mais pessoas em situação de rua têm taxa de ocupação **bem menor**.
- 🛌 Mais **quartos por residência** estão associados a **maior** ocupação.
- 🌊 A **presença de um rio** no bairro tem efeito positivo, porém pequeno.
- 🚨🏭 **Criminalidade** e **poluição** reduzem a ocupação.
- 📍 Bairros **mais distantes do centro** tendem a ter ocupação menor.
- 🧐 **`taxa_professores`** teve sinal negativo, o que é contraintuitivo. Vale investigar como essa variável foi medida (pode ser uma razão aluno/professor, por exemplo).
- 🗑️ **Variáveis descartadas** por multicolinearidade ou falta de significância: `index_acessibilidade`, `imposto_residencial`, `idade_media_residencias`, `proporcao_empresas`, `consumo_medio_energia` e `proporcao_lotes`.

### 📉 Qualidade geral

- O modelo explica cerca de **79% da variação** da taxa de ocupação (log) nos dados de treino.
- F-statistic = **208,3** (p ≈ 7e-128), então o modelo é globalmente significativo.
- 4 das 5 suposições foram plenamente satisfeitas.

---

## ⚠️ Limitações

- ❌ **Normalidade dos resíduos não foi satisfeita** (Shapiro-Wilk), embora o QQ-Plot mostre um desvio pequeno. Isso pode afetar a precisão de intervalos de confiança e valores-p.
- 🧪 Os dados de **teste foram preparados**, mas o modelo **ainda não foi avaliado neles**. Os resultados acima são de treino.
- 🔗 Correlação não é causalidade. O modelo mostra **associações**, não relações de causa e efeito.
- 📦 Amostra com apenas 500 bairros.

---

## 🚀 Próximos passos

- [ ] 🧪 Avaliar o modelo final no **conjunto de teste** (R², RMSE, MAE).
- [ ] 🔁 Aplicar **validação cruzada**.
- [ ] 🧹 Tratar outliers e testar novamente a normalidade dos resíduos.
- [ ] 🔄 Testar outras transformações e **regressão robusta** ou **mínimos quadrados ponderados**.
- [ ] 🌳 Comparar com modelos de Machine Learning (Random Forest, Gradient Boosting).
- [ ] 🔎 Investigar o sinal de `taxa_professores`.

---

## 🛠️ Como executar

**1. Clone o repositório**
```bash
git clone https://github.com/<seu-usuario>/<nome-do-repositorio>.git
cd <nome-do-repositorio>
```

**2. Instale as dependências**
```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn jupyter
```

**3. Coloque o arquivo `dataset.csv` na mesma pasta do notebook** 📂

**4. Abra o notebook**
```bash
jupyter notebook projeto.ipynb
```

---

## 🧰 Tecnologias

| Biblioteca | Uso |
|---|---|
| `pandas` / `numpy` | Manipulação de dados |
| `matplotlib` / `seaborn` | Visualização |
| `scipy` | Correlação de Pearson, Shapiro-Wilk, QQ-Plot |
| `statsmodels` | Regressão OLS, VIF, Rainbow, Durbin-Watson, Goldfeld-Quandt |
| `scikit-learn` | Divisão treino/teste e padronização |

---

## 👤 Autor

Feito com 💜 por **<seu-nome>**

🔗 [LinkedIn](https://linkedin.com/in/<seu-usuario>) • 🐙 [GitHub](https://github.com/<seu-usuario>)

⭐ Se este projeto te ajudou, deixe uma estrela no repositório!
