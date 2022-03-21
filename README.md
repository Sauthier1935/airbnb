# Airbnb - Preverndo a nota dos imóveis e classificando preços 

Problema: Prever a nota que um imóvel do airbnb possui a partir dos dados de imóveis do rio de janeiro 
Bônus: Categorizamos o atributo 'price' (preço) para o atributo resposta 'categorical_price' (preço categórico)
para prever uma categoria de preços de acordo com o dataset

1 - Dataset: https://www.kaggle.com/datasets/allanbruno/airbnb-rio-de-janeiro

- Temos um script de leitura de todos CSV, concatenando em um único datraframe. O dataset possui 108 colunas (108 atributos) e 840mil linhas (840mil instâncias)

- Percebendo a quatidade de atributos, em uma breve análise podemos perceber que muitos atributos não iriam nos ajudar no problema, portanto selecionamos 26 atributos destes 108.

Os atributos selecionados até aqui foram: 
'host_listings_count','latitude','longitude','property_type',
'room_type','accommodates','bathrooms','bedrooms','beds','bed_type',
'price','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights',
'number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness',
'review_scores_checkin','review_scores_communication','review_scores_location',
'review_scores_value','year','month'

Sendo 'review_scores_value' o atributo alvo para categorizar os imóveis de acordo com a sua nota (nota neste caso se refere a qualidade do imóvel, como as estrelas do Uber). Esta categoria apresenta 3 valores de nota para os imóveis no nosso trabalho. Notas: 10,9,8; ou seja as avalições de cada imóvel variam entre estes valores.

Após, alteramos o tipo de 3 dados: 'price','extra_people','cleaning_fee'. Estes dados são numéricos, porém o dataset trouxe eles como texto.

2- Análise de cada atributo:

A partir desta etapa, percebeu-se que temos 3 tipos de dados: contínuos, discretos e de texto
Assim, realizamos uma análise mais detalhada para cada tipo desses dados:

Dados contínuos:  'cleaning_fee', 'price','extra_people','review_scores_rating'
Dados discretos: 'accommodates','host_listings_count','bathrooms','bedrooms','beds',
                'minimum_nights','number_of_reviews','review_scores_value', 'month', 'year'
Dados em texto: 'property_type','room_type'

Com esta análise, percebemos que 1; alguns atributos previamente selecionados não fariam sentido e 2; precisávamos remover outliers, tratar textos etc.
1; Removemos os atributos: 'review_scores_cleanliness','review_scores_checkin', 'review_scores_communication',
       'review_scores_location','review_scores_accuracy','guests_included','maximum_nights','bed_type'
O motivo dessa remoção é que alguns atributos apresentavam pouquíssimos dados válidos (não nulos) ou possuíam diferença entre eles, ou seja, alguns desses atributos estavam preenchidos com o mesmo valor.

2; Para remover os outliers e normalizar esses dados, utilizamos uma técnica que define os quartis de cada atributo.
Na estatística descritiva, um quartil representa 1/4 da amostra ou população. Para nosso caso, 1/4 do atributo selecionado

Assim:
primeiro quartil (designado por Q1/4) = quartil inferior = é o valor aos 25% da amostra ordenada = 25º percentil
segundo quartil (designado por Q2/4) = mediana = é o valor até ao qual se encontra 50% da amostra ordenada = 50º percentil, ou 5º decil.
terceiro quartil (designado por Q3/4) = quartil superior = valor a partir do qual se encontram 25% dos valores mais elevados = valor aos 75% da amostra ordenada = 75º percentil

A partir desta técnica, determinamos pontos de outliers para cada atributo, calculando se cada valor do atributo estava inserido dentro dos limites inferiore e superior.

Estes limites são calculados a partir das seguintes equações:
   IQR = Q3-Q1
   Limite Superior = média do atributo + 1,5*IQR
   Limite Inferior = média do atributo - 1,5*IQR

Caso o valor estivesse fora da faixa entre Limite inferior e Limite superior, ele é considerado como outlier, sendo excluído do dataset.
Fonte: https://aprendendogestao.com.br/2016/08/26/identificacao-de-outliers/

Nesta etapa, ao executar o código, o script irá mostrar o comportamento de cada atributo, remover os outliers e após mostrar como se comportam após a remoção.

Ainda, para o dados de texto, observamos em gráficos que ambos atributos eram categóricos. Assim, tendo uma categoria com um conjunto de dados muito superior em relação aos demais. A solução para isso foi unir todas as categorias com poucos dados para uma única categoria chamada 'outros'. Por exemplo, para os tipos de imóveis, partimos de várias categorias, mas o resultado são as categorias Apartamento e Outros.

Finalmente, removemos todas as intâncias que apresentavam ainda algum dado nulo. 
Entendemos que assim teríamos nosso dataset puro, sem qualquer método de normalização. 
Como resultado, ficamos com 17 atributos e 180mil instâncias.

Após, criamos um mapa de calor para retirar algumas perguntas desse dataset, tirar insights e observar nosso atributo resposta 'review_scores_value'. 

3 - Seleção dos modelos, hyperparâmetros e gridsearch:

Os modelos selecionados para o problema foram SVM (máquinas de suporte vetorial), RF (Florestas Aleatórias), GNB (Modelo Gaussiano de Bayes) e KNN (K-Vizinhos Mais Próximos).

Para obter modelos que encontrassem melhores métricas de avaliação, utilizamos a técnica gridsearch onde definimos hyperparâmetros (uma lista de parâmetros) para cada modelo que este realiza vários testes com estes parâmetros apresentando a configuração de parâmetros que traz a melhor acurácia. 
Outro parâmetro definido foi o cross-validation que foi setado para o valor 10.

Para cada modelo tais hyperparâmetros foram definidos:
SVM - {'kernel':['rbf','poly'], 'C':[1,100,500]}
RF - {'criterion':['gini','entropy'],'max_depth': [80,95,100],'n_estimators': [100,250,500,1000]}
GNB - {'priors': [None,1,10,20],'var_smoothing':[1, 3.16227766e-05, 1.00000000e-09]}
KNN - {'n_neighbors': [5,10,20,50,100]}

A partir destes hiperparâmetros a técnica gridsearch retornou as melhores configurações:
SVM - {'C': 500, 'kernel': 'poly'}
RF - {'criterion': 'gini', 'max_depth': 80, 'n_estimators': 100}
GNB - {'priors': None, 'var_smoothing': 1e-09}
KNN - {'n_neighbors': 5}

4- Modelos preditivos:

Para aplicação dos modelos preditivos, optamos por realizar uma amostragem menor dos dados obtidos. Isto porque os modelos estavam levando muito tempo para realizar os cálculos. A partir de 184mil instâncias, optamos por utilizar 35mil instâncias. 
No entanto, ao retirar uma amostra desses dados, para evitar problemas de balanceamento do dataset, utilizamos o método SMOTE, nos retornando 63450 instâncias. Finalmente, esta quantidade de instâncias balanceadas foi utilizada para a aplicação dos modelos preditivos.

Com estas configurações, criamos os modelos preditivos que retornaram as seguintes acurácias, matriz de confusão:

SVM: 60.45547675334909
RF: 93.50827423167848
GNB: 55.56973995271868
KNN: 83.69582348305752

Bônus:

Observando também outro problema, os preços dos imóveis, decidimos criar outra categoria chamada 'categoriacal_price'. 
Esta categoria separa os valores dos preços dos imóveis em 4, utilizando o método de cluesterização k-means. Entendemos que para fins de aprendizado, utilizar o k-menas para um único atributo e categorizar se fez importante obtendo os seguintes resultados:

Centroides de cada Categoria:

0 - R$ 411.278992
1 - R$ 114.134003
2 - R$ 673.961975
3 - R$ 237.809998

Intervalo de cada Categoria:

0 - R$ 325.0 < x < R$ 542.0
1 - R$ 0.000 < x < R$ 175.0
2 - R$ 176.0 < x < R$ 324.0
3 - R$ 543.0 < x < R$ 905.0

Assim, novamente obtivemos o mapa de calor para obter insights e criamos um mapa iterativo do rio de janeiro para olhar as localizações onde os imóveis se mostraram mais caros.

Para cada modelo tais hyperparâmetros foram definidos:
SVM - {'kernel':['rbf','poly'], 'C':[1,100,500]}
RF - {'criterion':['gini','entropy'],'max_depth': [80,95,100],'n_estimators': [100,250,500,1000]}
GNB - {'priors': [None,1,10,20],'var_smoothing':[1, 3.16227766e-05, 1.00000000e-09]}
KNN - {'n_neighbors': [5,10,20,50,100]}

A partir destes hiperparâmetros a técnica gridsearch retornou as melhores configurações:
SVM - {'C': 500, 'kernel': 'rbf'}
RF - {'criterion': 'gini', 'max_depth': 80, 'n_estimators': 500}
GNB - {'priors': None, 'var_smoothing': 1e-09}
KNN - {'n_neighbors': 10}


Com estas configurações, criamos os modelos preditivos que retornaram as seguintes acurácias, matriz de confusão:

SVM: 31.96588809712325
RF: 62.88763525996305
GNB: 32.66363156505675
KNN: 55.092042755344416

Conclusões:

Não basta ter um grande volume de dados, precisamos ter qualidade nesse conjunto. Mesmo partindode mais de 840 mil instâncias, nosso dataset pós preparação, possuía 184 mil instâncias. Mesmo assim, para fins de conclusão do trabalho usamos apenas 35 mil, e após balanceamos suas classes.
Uma análise profunda de cada atributo permite realizar uma preparação que irá impactar na qualidade do modelo, onde utilizamos uma técnica para remover outliers.
Para o conjunto de dados, o modelo RF apresentou o maior score para ambas classificações, podendo ser observado na segunda predição que a classe dominante em RF foi a classe 1 (valores dos imóveis mais baratos). Essa conclusão pode ser obtida a partir das matrizes de confusão criadas.

Melhorias:

Utilizar métricas de validação - pode-se melhorar o preparo das instâncias e atributos visualizando previamente os dados.
Utilizar outras técnicas de teste e treino de dados - para validar os modelos, poderíamos tirar outras conclusões usando outras técnicas.
Utilizar outros modelos preditivos - buscando melhor assetividade, podemos sugerir usar redes neurais para o problema
Utilizar outras métricas de avaliação dos modelos - podendo assim observar outras características nos resultados de acordo com o problema
