from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
class Models:

    def grid_search(parameters, model, cv_layers, score,total_data,target):
        best = GridSearchCV(model, parameters, cv=cv_layers, scoring=score, return_train_score=False, verbose=1)
        best.fit(total_data, target)
        return best.best_params_

    def k_means():
        return KMeans()

    def support_vector_machine():
        return svm.SVC()

    def randon_forest():
        return RandomForestClassifier()

    def gaussian_naive_bayes():
        return GaussianNB()

    def k_neighbors():
        return KNeighborsClassifier()

    def model_k_means(algorithm, max_iter, n_clusters):
        model = KMeans(algorithm = algorithm, max_iter = max_iter, n_clusters = n_clusters)
        return model

    def model_svm(C, kernel):
        model = svm.SVC(C=C, kernel=kernel)
        return model

    def model_rf(criterion,max_depth, n_estimators):
        model = RandomForestClassifier(criterion = criterion, max_depth = max_depth, n_estimators = n_estimators)
        return model

    def model_gnb(priors, var_smoothing):
        model = GaussianNB(priors = priors, var_smoothing = var_smoothing)
        return model

    def model_knn(n_neighbors):
        model = KNeighborsClassifier(n_neighbors = n_neighbors)
        return model

    def model_to_predict(model,dfx,dfy,cv_layers):
        predictions = cross_val_predict(model, dfx, dfy, cv=cv_layers)
        return predictions

    def accuracy(dfy, predictions):
        score = accuracy_score(dfy,predictions)*100
        return score

    def model_confusion_matrix(dfy,predictions):
        cf = confusion_matrix(dfy, predictions)
        return cf
