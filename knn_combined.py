# Imports
import time
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from utility1 import load_data, plot_learning_curves, report, plot_bars, plot_lines

t0 = time.time()

X_train1_m, X_test_m, y_train1_m, y_test_m, class_names_m = load_data('motions')
X_train1_p, X_test_p, y_train1_p, y_test_p, class_names_p = load_data('particles')
col_names_p = X_train1_p.columns

X_val_m, X_train_m, y_val_m, y_train_m = train_test_split(X_train1_m, y_train1_m, test_size = 0.7, random_state=42)
X_val_p, X_train_p, y_val_p, y_train_p = train_test_split(X_train1_p, y_train1_p, test_size = 0.7, random_state=42)

########## BEST FOUND PARAMETERS - Motions #############
best_k_m = 7
best_metric_m = 'minkowski'
best_p_m = 2 #i.e. euclidean distance
best_algorithm_m = 'auto'
########################################################

########## BEST FOUND PARAMETERS - Particles ###########
best_k_p = 7
best_metric_p = 'minkowski'
best_p_p = 1 #i.e. manhattan distance
best_algorithm_p = 'brute'
########################################################


####BASE MODEL###
model_m = KNeighborsClassifier(n_neighbors = best_k_m, metric = best_metric_m, algorithm=best_algorithm_m, p = best_p_m)
model_p = KNeighborsClassifier(n_neighbors = best_k_p, metric = best_metric_p, algorithm=best_algorithm_p, p = best_p_p)
#####

#Data scaling
scale = True
if scale:
    scaler_m = StandardScaler()
    scaler_m.fit(X_train1_m)
    
    X_train1_m = scaler_m.transform(X_train1_m)
    X_train_m = scaler_m.transform(X_train_m)
    X_val_m = scaler_m.transform(X_val_m)
    X_test_m = scaler_m.transform(X_test_m)
    
    scaler_p = StandardScaler()
    scaler_p.fit(X_train1_p)
    
    X_train1_p = scaler_p.transform(X_train1_p)
    X_train_p = scaler_p.transform(X_train_p)
    X_val_p = scaler_p.transform(X_val_p)
    X_test_p = scaler_p.transform(X_test_p)

scorer = make_scorer(cohen_kappa_score)

useRandomCV = False
if useRandomCV:
    
    #switch between motions and particle sets for RandomizedSearchCV
    test_motions = False
    if test_motions:
        model = model_m
        X_train = X_train_m
        y_train = y_train_m
        print("RandomizedSearchCV testing the Motions dataset")
    else:
        model = model_p
        X_train = X_train_p
        y_train = y_train_p
        print("RandomizedSearchCV testing the Particles dataset")
              
    # specify parameters and distributions to sample from
    param_dist = {"n_neighbors": range(1, 200, 1),
              "metric": ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'seuclidean'],
              "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
              "p": [2**(i/4) for i in range(0, 16)]
              #n_neighbors = n, metric = best_metric_m, algorithm=best_algorithm_m, p = best_p_m
              }

    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
    n_iter_search = 1000
    random_search = RandomizedSearchCV(model, param_distributions = param_dist, n_iter=n_iter_search, cv = 5, scoring='accuracy', verbose=10)    
    random_search.fit(X_train, y_train)
    report(random_search.cv_results_)
    #scores = random_search.cv_results_['mean_test_score']


grid_cv = False
if grid_cv:
    
    #switch between motions and particle sets for RandomizedSearchCV
    test_motions = True
    if test_motions:
        model = model_m
        X_train = X_train_m
        y_train = y_train_m
        print("RandomizedSearchCV testing the Motions dataset")
    else:
        model = model_p
        X_train = X_train_p
        y_train = y_train_p
        print("RandomizedSearchCV testing the Particles dataset")
    
    parameters= {"n_neighbors": range(1, 200, 1),
                 "metric": ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'seuclidean'], 
                 "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                 "p": [2**(i/4) for i in range(0, 16)]
                 }
    clf = GridSearchCV(model, parameters, cv=5)
    clf.fit(X=X_train, y=y_train)
    best_model = clf.best_estimator_
    print(clf.best_score_, clf.best_params_)

test_k = False
if test_k:
    print('heya')
    test_parameter = 'Best k-value'
    n_range = range(1, 200, 1)
    
    scores_m = {}
    scores_list_m = []
    time_list_m = []
    
    scores_p = {}
    scores_list_p = []
    time_list_p = []
    
    for n in n_range:
        # Motions
        t1 = time.time()
        print("looking at {} = {} on Motions Set".format(test_parameter, n))
        model_m = KNeighborsClassifier(n_neighbors = n, metric = best_metric_m, algorithm=best_algorithm_m, p = best_p_m)
        model_m.fit(X_train_m, y_train_m.values.ravel('C'))
        y_pred_m = model_m.predict(X_val_m)
        scores_m[n] = accuracy_score(y_val_m, y_pred_m)
        scores_list_m.append(scores_m[n])
        print("took {} seconds".format(time.time()-t1))
        time_list_m.append(time.time()-t1)
        
        #Particles
        t1 = time.time()
        print("looking at {} = {} on Particles Set".format(test_parameter, n))
        model_p = KNeighborsClassifier(n_neighbors = n, metric = best_metric_p, algorithm=best_algorithm_p, p = best_p_p)
        model_p.fit(X_train_p, y_train_p.values.ravel('C'))
        y_pred_p = model_p.predict(X_val_p)
        scores_p[n] = accuracy_score(y_val_p, y_pred_p)
        scores_list_p.append(scores_p[n])
        print("took {} seconds".format(time.time()-t1))
        time_list_p.append(time.time()-t1)
        
    # matplotlib is clunky in trying to plot bars side by side, BUT
    plot_lines(scores_list_m, time_list_m, scores_list_p, time_list_p, test_parameter, n_range)
    
    
    print("{} for best validation set accuracy on Motions: {}".format(test_parameter, max(scores_m, key=scores_m.get)))
    print("{} for best validation set accuracy on Particles: {}".format(test_parameter, max(scores_p, key=scores_p.get)))
    
test_metric = False
if test_metric:
    test_parameter = 'Metric'
    n_range = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'seuclidean']
    
    scores_m = {}
    scores_list_m = []
    time_list_m = []
    
    scores_p = {}
    scores_list_p = []
    time_list_p = []
    
    for n in n_range:
        # Motions
        t1 = time.time()
        print("looking at {} = {} on Motions Set".format(test_parameter, n))
        model_m = KNeighborsClassifier(n_neighbors = best_k_m, metric = n, algorithm=best_algorithm_m, p = best_p_m)
        model_m.fit(X_train_m, y_train_m.values.ravel('C'))
        y_pred_m = model_m.predict(X_val_m)
        scores_m[n] = accuracy_score(y_val_m, y_pred_m)
        scores_list_m.append(scores_m[n])
        print("took {} seconds".format(time.time()-t1))
        time_list_m.append(time.time()-t1)
        
        #Particles
        t1 = time.time()
        print("looking at {} = {} on Particles Set".format(test_parameter, n))
        model_p = KNeighborsClassifier(n_neighbors = best_k_p, metric =n, algorithm=best_algorithm_p, p = best_p_p)
        model_p.fit(X_train_p, y_train_p.values.ravel('C'))
        y_pred_p = model_p.predict(X_val_p)
        scores_p[n] = accuracy_score(y_val_p, y_pred_p)
        scores_list_p.append(scores_p[n])
        print("took {} seconds".format(time.time()-t1))
        time_list_p.append(time.time()-t1)
        
    # matplotlib is clunky in trying to plot bars side by side, BUT
    plot_bars(scores_list_m, time_list_m, scores_list_p, time_list_p, test_parameter, n_range)
    
    print("{} for best validation set accuracy on Motions: {}".format(test_parameter, max(scores_m, key=scores_m.get)))
    print("{} for best validation set accuracy on Particles: {}".format(test_parameter, max(scores_p, key=scores_p.get)))
    
test_p = False
if test_p:
    test_parameter = 'minkowski p-value'
    n_range = [2**(i/4) for i in range(0, 16)]
    
    scores_m = {}
    scores_list_m = []
    time_list_m = []
    
    scores_p = {}
    scores_list_p = []
    time_list_p = []
    
    for n in n_range:
        # Motions
        t1 = time.time()
        print("looking at {} = {} on Motions Set".format(test_parameter, n))
        model_m = KNeighborsClassifier(n_neighbors = best_k_m, metric = 'minkowski', algorithm=best_algorithm_m, p = n)
        model_m.fit(X_train_m, y_train_m.values.ravel('C'))
        y_pred_m = model_m.predict(X_val_m)
        scores_m[n] = accuracy_score(y_val_m, y_pred_m)
        scores_list_m.append(scores_m[n])
        print("took {} seconds".format(time.time()-t1))
        time_list_m.append(time.time()-t1)
        
        #Particles
        t1 = time.time()
        print("looking at {} = {} on Particles Set".format(test_parameter, n))
        model_p = KNeighborsClassifier(n_neighbors = best_k_p, metric = 'minkowski', algorithm=best_algorithm_p, p = n)
        model_p.fit(X_train_p, y_train_p.values.ravel('C'))
        y_pred_p = model_p.predict(X_val_p)
        scores_p[n] = accuracy_score(y_val_p, y_pred_p)
        scores_list_p.append(scores_p[n])
        print("took {} seconds".format(time.time()-t1))
        time_list_p.append(time.time()-t1)
        
    # matplotlib is clunky in trying to plot bars side by side, BUT
    plot_lines(scores_list_m, time_list_m, scores_list_p, time_list_p, test_parameter, n_range)
    
    print("{} for best validation set accuracy on Motions: {}".format(test_parameter, max(scores_m, key=scores_m.get)))
    print("{} for best validation set accuracy on Particles: {}".format(test_parameter, max(scores_p, key=scores_p.get)))
    
    
test_algorithm = False
if test_algorithm:
    test_parameter = 'Algorithm'
    n_range = ['ball_tree', 'kd_tree', 'brute'] 
    
    scores_m = {}
    scores_list_m = []
    time_list_m = []
    
    scores_p = {}
    scores_list_p = []
    time_list_p = []
    
    for n in n_range:
        # Motions
        t1 = time.time()
        print("looking at {} = {} on Motions Set".format(test_parameter, n))
        model_m = KNeighborsClassifier(n_neighbors = best_k_m, metric = best_metric_m, algorithm=best_algorithm_m, p = best_p_m)
        model_m.fit(X_train_m, y_train_m.values.ravel('C'))
        y_pred_m = model_m.predict(X_val_m)
        scores_m[n] = accuracy_score(y_val_m, y_pred_m)
        scores_list_m.append(scores_m[n])
        print("took {} seconds".format(time.time()-t1))
        time_list_m.append(time.time()-t1)
        
        #Particles
        t1 = time.time()
        print("looking at {} = {} on Particles Set".format(test_parameter, n))
        model_p = KNeighborsClassifier(n_neighbors = best_k_p, metric = best_metric_p, algorithm=best_algorithm_p, p = best_p_p)
        model_p.fit(X_train_p, y_train_p.values.ravel('C'))
        y_pred_p = model_p.predict(X_val_p)
        scores_p[n] = accuracy_score(y_val_p, y_pred_p)
        scores_list_p.append(scores_p[n])
        print("took {} seconds".format(time.time()-t1))
        time_list_p.append(time.time()-t1)
        
    # matplotlib is clunky in trying to plot bars side by side, BUT
    plot_bars(scores_list_m, time_list_m, scores_list_p, time_list_p, test_parameter, n_range)
    
    print("{} for best validation set accuracy on Motions: {}".format(test_parameter, max(scores_m, key=scores_m.get)))
    print("{} for best validation set accuracy on Particles: {}".format(test_parameter, max(scores_p, key=scores_p.get)))


show_validation_performance = False
if show_validation_performance:
    
    #on Motions Set
    print("Motions Performance on the validation set:")
    best_m = KNeighborsClassifier(n_neighbors = best_k_m, metric = best_metric_m, algorithm=best_algorithm_m, p = best_p_m)
    t_fit1_m = time.time()
    best_m.fit(X_train_m, y_train_m.values.ravel('C'))  #train on the whole training set
    print("Motions fitting time = {}".format(time.time()-t_fit1_m))
    t_pred1_m = time.time()
    y_pred_m = best_m.predict(X_val_m)
    print("Motions validation set prediction time = {}".format(time.time()-t_pred1_m))
    
    #on Particles Set
    print("Particles Performance on the validation set:")
    best_p = KNeighborsClassifier(n_neighbors = best_k_p, metric = best_metric_p, algorithm=best_algorithm_p, p = best_p_p)
    t_fit1_p = time.time()
    best_p.fit(X_train_p, y_train_p.values.ravel('C'))  #train on the whole training set
    print("Motions fitting time = {}".format(time.time()-t_fit1_p))
    t_pred1_p = time.time()
    y_pred_p = best_p.predict(X_val_p)
    print("Motions validation set prediction time = {}".format(time.time()-t_pred1_p))
    
    #Performance Report, Motions
    for motion_type in class_names_m:
        pred_score_m = best_m.score(X_val_m[y_val_m.motion_type==motion_type], y_val_m[y_val_m.motion_type==motion_type])
        print("{} accuracy = {p:8.4f}".format(motion_type, p=pred_score_m))
    print("Cohen Kappa: {}".format(cohen_kappa_score(y_pred_m, y_val_m)))   
    print("Accuracy: {}".format(accuracy_score(y_pred_m, y_val_m)))
    print("F1 Score: {}".format(f1_score(y_pred_m, y_val_m, average = 'weighted')))
    print("Precision: {}".format(precision_score(y_pred_m, y_val_m, average='weighted')))
    print("Recall: {}".format(recall_score(y_pred_m, y_val_m, average='weighted')))    
    
    #Performance Report, Motions
    for particle_type in class_names_p:
        pred_score_p = best_p.score(X_val_p[y_val_p.id==particle_type], y_val_p[y_val_p.id==particle_type])
        print("{} accuracy = {p:8.4f}".format(particle_type, p=pred_score_p))
    print("Cohen Kappa: {}".format(cohen_kappa_score(y_pred_p, y_val_p)))   
    print("Accuracy: {}".format(accuracy_score(y_pred_p, y_val_p)))
    print("F1 Score: {}".format(f1_score(y_pred_p, y_val_p, average = 'weighted')))
    print("Precision: {}".format(precision_score(y_pred_p, y_val_p, average='weighted')))
    print("Recall: {}".format(recall_score(y_pred_p, y_val_p, average='weighted')))    


finalModel = True
if finalModel:

    #on Motions Set
    print("Motions Performance on the test set:")
    best_m = KNeighborsClassifier(n_neighbors = best_k_m, metric = best_metric_m, algorithm=best_algorithm_m, p = best_p_m)
    t_fit1_m = time.time()
    best_m.fit(X_train_m, y_train_m.values.ravel('C'))  #train on the whole training set
    print("Motions fitting time = {}".format(time.time()-t_fit1_m))
    t_pred1_m = time.time()
    y_pred_m = best_m.predict(X_test_m)
    print("Motions test set prediction time = {}".format(time.time()-t_pred1_m))

    #on Particles Set
    print("Particles Performance on the test set:")
    best_p = KNeighborsClassifier(n_neighbors = best_k_p, metric = best_metric_p, algorithm=best_algorithm_p, p = best_p_p)
    t_fit1_p = time.time()
    best_p.fit(X_train_p, y_train_p.values.ravel('C'))  #train on the whole training set
    print("Particles fitting time = {}".format(time.time()-t_fit1_p))
    t_pred1_p = time.time()
    y_pred_p = best_p.predict(X_test_p)
    print("Particles test set prediction time = {}".format(time.time()-t_pred1_p))
    
    #Performance Report, Motions
    for motion_type in class_names_m:
        pred_score_m = best_m.score(X_test_m[y_test_m.motion_type==motion_type], y_test_m[y_test_m.motion_type==motion_type])
        print("{} accuracy = {p:8.4f}".format(motion_type, p=pred_score_m))
    print("Cohen Kappa: {}".format(cohen_kappa_score(y_pred_m, y_test_m)))   
    print("Accuracy: {}".format(accuracy_score(y_pred_m, y_test_m)))
    print("F1 Score: {}".format(f1_score(y_pred_m, y_test_m, average = 'weighted')))
    print("Precision: {}".format(precision_score(y_pred_m, y_test_m, average='weighted')))
    print("Recall: {}".format(recall_score(y_pred_m, y_test_m, average='weighted')))    
    
    #Performance Report, Motions
    for particle_type in class_names_p:
        pred_score_p = best_p.score(X_test_p[y_test_p.id==particle_type], y_test_p[y_test_p.id==particle_type])
        print("{} accuracy = {p:8.4f}".format(particle_type, p=pred_score_p))
    print("Cohen Kappa: {}".format(cohen_kappa_score(y_pred_p, y_test_p)))   
    print("Accuracy: {}".format(accuracy_score(y_pred_p, y_test_p)))
    print("F1 Score: {}".format(f1_score(y_pred_p, y_test_p, average = 'weighted')))
    print("Precision: {}".format(precision_score(y_pred_p, y_test_p, average='weighted')))
    print("Recall: {}".format(recall_score(y_pred_p, y_test_p, average='weighted')))    


learning_curves = False
if learning_curves:
    plot_learning_curves(model_m, X_train1_m, y_train1_m, low_limit=0.5, title = "SVM - Motions Set - Post-Tuning Learning Curves")
    plot_learning_curves(model_p, X_train1_p, y_train1_p, low_limit=0.5, title = "SVM - Particles Set - Post-Tuning Learning Curves")
    
    
print("time elapsed: {}".format(time.time()-t0))




# References
# https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html