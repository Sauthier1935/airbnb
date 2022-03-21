import models


class best_parms:

    def search_best_parms(cv_layers,score,total_data,target_y):

        # Support Vector Machine
        parameters = {'kernel':['rbf','poly'], 'C':[1,100,500]}
        model = models.Models.support_vector_machine()
        params = models.Models.grid_search(parameters,model,cv_layers,score,total_data,target_y)
        print(params)
        ##resultado: {'C': 500, 'kernel': 'poly'}

        # Randon Forest
        parameters = {'criterion':['gini','entropy'],'max_depth': [80,95,100],
                      'n_estimators': [100,250,500,1000]}
        model = models.Models.randon_forest()
        params = models.Models.grid_search(parameters,model,cv_layers,score,total_data,target_y)
        print(params)
        #resultado: {'criterion': 'gini', 'max_depth': 80, 'n_estimators': 100}

        # Gaussian Naive Bayes
        parameters = {'priors': [None,1,10,20],
                      'var_smoothing':[1, 3.16227766e-05, 1.00000000e-09]}
        model = models.Models.gaussian_naive_bayes()
        params = models.Models.grid_search(parameters,model,cv_layers,score,total_data,target_y)
        print(params)
        #resultado: {'priors': None, 'var_smoothing': 1e-09}

        # K Vizinhos Mais Próximos
        parameters = {'n_neighbors': [5,10,20,50,100]}
        model = models.Models.k_neighbors()
        params = models.Models.grid_search(parameters,model,cv_layers,score,total_data,target_y)
        print(params)
        #resultado: {'n_neighbors': 5}