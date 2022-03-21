import pandas as pd
pd.set_option('display.max_columns', None)

# Importação de outros scripts
import read_files
import utils
import graphics
import models
import search_best_params

# Retirar mensagens de aviso
import warnings
warnings.simplefilter("ignore")

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Ler os arquivos com dados
df_airbnb = read_files.read()
#df_airbnb.head(1000).to_csv(r'C:\Users\sauth\PycharmProjects\main\dados.csv')
# Tratamentos
# - Como existem muitos atributos, a ideia é analisar os atributos que realmente importam para o problema
# - Uma análise rápida pode permitir ver os atributos que não serão necessários para os modelos de previsão:
# - Tipos de atributos que serão excluídos:
# - IDs, Links e informações não relevantes para prever o preço
# - Atributos repetidos ou extremamente parecidas com outra (que dão a mesma informação)
# - Atributos com textos e listas
# - Atributos em que todos ou quase todos os valores são iguais ou nulos
# - Após da análise qualitativa de cada atributo, levando em conta os critérios explicados acima,
#   ficamos com os seguintes, abaixo

col = ['host_listings_count','latitude','longitude','property_type',
       'room_type','accommodates','bathrooms','bedrooms','beds','bed_type',
       'price','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights',
       'number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness',
       'review_scores_checkin','review_scores_communication','review_scores_location',
       'review_scores_value','year','month']

# no total eram 108 atributos, reduzindo para 26

# Pré Processamento de dados
# - Retirar dados faltantes
# - Tratar tipo de atributo para atributo numérico
# - Nomear atributo resposta
# - Retirar um número de amostras satisfatório para o problema
df_airbnb = df_airbnb.loc[:, col]

# Tipos de cada atributo
utils.Utils.print_df_types(df_airbnb)

# Troca tipo dos atributos
col = ['price','extra_people','cleaning_fee']
for c in col:
       df_airbnb[c] = utils.Utils.str_attributes_to_int(df_airbnb[c])

# Analisar cada coluna (se necessário)
# Retirar atributos entendidas como irrelevantes para o problema
# (este passo foi adicionado após observar os gráficos)
# os atributos selecionadas aqui não trazem nenhuma informação com maiores diferenças
# guests_included - a maior parte dos usuários preenche o valor dessa variável com valor 1
# maximum_nights - a maior parte dos imóveis são preenchidos com valor zero e
#                  poucos são preenchidos com valores diferentes
# os atributos selecionadas aqui não trazem nenhuma informação com maiores diferenças


# Atributos com valores contínuos a serem tratados
col = ['cleaning_fee', 'price','extra_people','review_scores_rating']
for c in col:
       df_airbnb = utils.Utils.analysis_continuous_attributes(df_airbnb, c)

# Atributos com valores discretos a serem tratados
col = ['accommodates','host_listings_count','bathrooms','bedrooms','beds',
       'minimum_nights','number_of_reviews','review_scores_value']
for c in col:
       df_airbnb = utils.Utils.analysis_discrete_attributes(df_airbnb,c)

# Atributos com valores de texto
col = ['property_type','room_type']
for c in col:
       df_airbnb = utils.Utils.analysis_text_attributes(df_airbnb, c, 50000)
       # Alterar para valores numéricos
       df_airbnb = utils.Utils.labelencoder__attributes_to_int(df_airbnb,c)

col = ['review_scores_cleanliness','review_scores_checkin', 'review_scores_communication',
       'review_scores_location','review_scores_accuracy','guests_included','maximum_nights','bed_type']
for c in col:
       df_airbnb = df_airbnb.drop(c, axis=1)

df_airbnb = df_airbnb.dropna()

# Tipos de cada atributo
utils.Utils.print_df_types(df_airbnb)

# Prever a nota do imóvel
# Hiper parâmetros - grid search
# Modelos Random Forest, SVM, Naive-Bayes, KNN
# Acurácia e métricas de avaliação matriz de confusão

# Dados tratados e prontos
n_split = 35000
df_search_best_parms = utils.Utils.split_data(df_airbnb,n_split)

#heatmap
graphics.Graphics_utils.plot_heatmap(df_search_best_parms)

target_column = ['review_scores_value']
target_y, total_data = utils.Utils.selection(df_search_best_parms,target_column)
target_y = target_y.values
# Procurar Melhores parâmetros pra cada modelos
cv_layers = 10
score = 'accuracy'

# Função Gridsearch
search_best_params.best_parms.search_best_parms(cv_layers,score,total_data,target_y)

# target
n_split = 35000
df_search_best_parms = utils.Utils.split_data(df_airbnb,n_split)
dfy, dfx = utils.Utils.preparation(df_search_best_parms,target_column)

## parametros
##svm_params = {'C': 500, 'kernel': 'poly'}
##rf_params = {'criterion': 'gini', 'max_depth': 80, 'n_estimators': 100}
##gnb_params = {'priors': None, 'var_smoothing': 1e-09}
##knn_params = {'n_neighbors': 5}

## Models
svm_model = models.Models.model_svm(500,'poly')
rf_model = models.Models.model_rf('gini', 80, 100)
gnb_model = models.Models.model_gnb(None, 1e-09)
knn_model = models.Models.model_knn(5)
#
## Predictions
pred_svm = models.Models.model_to_predict(svm_model,dfx,dfy,cv_layers)
pred_rf = models.Models.model_to_predict(rf_model,dfx,dfy,cv_layers)
pred_gnb = models.Models.model_to_predict(gnb_model,dfx,dfy,cv_layers)
pred_knn = models.Models.model_to_predict(knn_model,dfx,dfy,cv_layers)
#
## Score
print('SVM: {}'.format(models.Models.accuracy(dfy,pred_svm)))
print('RF: {}'.format(models.Models.accuracy(dfy,pred_rf)))
print('GNB: {}'.format(models.Models.accuracy(dfy,pred_gnb)))
print('KNN: {}'.format(models.Models.accuracy(dfy,pred_knn)))
#
## Matriz de confusão
cf_svm = models.Models.model_confusion_matrix(dfy,pred_svm)
cf_rf = models.Models.model_confusion_matrix(dfy,pred_rf)
cf_gnb = models.Models.model_confusion_matrix(dfy,pred_gnb)
cf_knn = models.Models.model_confusion_matrix(dfy,pred_knn)
#
## Plot matriz de confusão
lbl1 = [8, 9, 10]
lbl2 = [8, 9, 10]
graphics.Graphics_utils.plot_confusion_matrix(cf_svm,lbl1,lbl2)
graphics.Graphics_utils.plot_confusion_matrix(cf_rf,lbl1,lbl2)
graphics.Graphics_utils.plot_confusion_matrix(cf_gnb,lbl1,lbl2)
graphics.Graphics_utils.plot_confusion_matrix(cf_knn,lbl1,lbl2)
#

#print mapa rio
graphics.Graphics_utils.plot_map(df_airbnb)


# Criando a coluna categórica de preço
target_column = ['price']
target_y, total_data = utils.Utils.selection(df_airbnb,target_column)

#Build clusters
graphics.Graphics_utils.build_clusters(target_y)

#Build with 4 clusters
clusters = KMeans(n_clusters=4, verbose=1)
clusters.fit(target_y)
target_y['categorical_price'] = clusters.labels_

clust_profile = pd.pivot_table(target_y, values=target_y.columns,index='categorical_price',aggfunc=np.mean)
print(np.round(clust_profile,3))

#Applying kmeans to the dataset / Creating the kmeans classifier
#random_state is the seed used by the random number generator.
#If random_state is None, the random number generator is the RandomState instance used by np.random.
kmeans = KMeans(n_clusters = 4, max_iter = 1000, random_state = 42)
y_kmeans = kmeans.fit_predict(target_y)

#Convert the x_train dataframe to a numpy array
x_train_arr = target_y.values

# Range dos valores de cada categoria
print('mín:{} max{}'.format(min(x_train_arr[y_kmeans == 0, 0]),max(x_train_arr[y_kmeans == 0, 0])))
print('mín:{} max{}'.format(min(x_train_arr[y_kmeans == 1, 0]),max(x_train_arr[y_kmeans == 1, 0])))
print('mín:{} max{}'.format(min(x_train_arr[y_kmeans == 2, 0]),max(x_train_arr[y_kmeans == 2, 0])))
print('mín:{} max{}'.format(min(x_train_arr[y_kmeans == 3, 0]),max(x_train_arr[y_kmeans == 3, 0])))

#Visualising the clusters and Plotting the centroids of the clusters
graphics.Graphics_utils.plot_clusters_k_means(x_train_arr,y_kmeans,kmeans)

# Adiciona variável no dataset
df_airbnb = df_airbnb.join(target_y['categorical_price'])

# Remove categoria preço contínuo
df_airbnb = df_airbnb.drop('price', axis=1)

#Mapa de calor
graphics.Graphics_utils.plot_heatmap(df_airbnb)

# target
# Procurar Melhores parâmetros pra cada modelos
cv_layers = 10
score = 'accuracy'
n_split = 35000
df_search_best_parms = utils.Utils.split_data(df_airbnb,n_split)
target_column = ['categorical_price']
target_y, total_data = utils.Utils.selection(df_search_best_parms,target_column)
search_best_params.best_parms.search_best_parms(cv_layers,score,total_data,target_y)

# target
n_split = 35000
df_search_best_parms = utils.Utils.split_data(df_airbnb,n_split)
dfy, dfx = utils.Utils.preparation(df_search_best_parms,target_column)

# parametros
#svm_params = {'C': 500, 'kernel': 'rbf'}
#rf_params = {'criterion': 'gini', 'max_depth': 80, 'n_estimators': 500}
#gnb_params = {'priors': None, 'var_smoothing': 1e-09}
#knn_params = {'n_neighbors': 10}

# Models
svm_model = models.Models.model_svm(500,'rbf')
rf_model = models.Models.model_rf('gini', 80, 500)
gnb_model = models.Models.model_gnb(None, 1e-09)
knn_model = models.Models.model_knn(10)

# Predictions
pred_svm = models.Models.model_to_predict(svm_model,dfx,dfy,cv_layers)
pred_rf = models.Models.model_to_predict(rf_model,dfx,dfy,cv_layers)
pred_gnb = models.Models.model_to_predict(gnb_model,dfx,dfy,cv_layers)
pred_knn = models.Models.model_to_predict(knn_model,dfx,dfy,cv_layers)

# Score
print('SVM: {}'.format(models.Models.accuracy(dfy,pred_svm)))
print('RF: {}'.format(models.Models.accuracy(dfy,pred_rf)))
print('GNB: {}'.format(models.Models.accuracy(dfy,pred_gnb)))
print('KNN: {}'.format(models.Models.accuracy(dfy,pred_knn)))

# Matriz de confusão
cf_svm = models.Models.model_confusion_matrix(dfy,pred_svm)
cf_rf = models.Models.model_confusion_matrix(dfy,pred_rf)
cf_gnb = models.Models.model_confusion_matrix(dfy,pred_gnb)
cf_knn = models.Models.model_confusion_matrix(dfy,pred_knn)

# Plot matriz de confusão
lbl1 = [0, 1, 2, 3]
lbl2 = [0, 1, 2, 3]
graphics.Graphics_utils.plot_confusion_matrix(cf_svm,lbl1,lbl2)
graphics.Graphics_utils.plot_confusion_matrix(cf_rf,lbl1,lbl2)
graphics.Graphics_utils.plot_confusion_matrix(cf_gnb,lbl1,lbl2)
graphics.Graphics_utils.plot_confusion_matrix(cf_knn,lbl1,lbl2)

