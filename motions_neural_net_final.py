# Imports
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.backend import clear_session
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint as sp_randint
from keras.layers import LeakyReLU
from sklearn.metrics import make_scorer, cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score
from utility1 import load_data, plot_learning_curves, report, plot_lines1

t0 = time.time()

# Randomly divide into train and test sets
X_train1, X_test, y_train1, y_test, class_names = load_data('motions')

# Explicitly get a validation set
X_val, X_train, y_val, y_train = train_test_split(X_train1, y_train1, test_size = 0.7, random_state=42)

# Scale
scaler = StandardScaler()
scaler.fit(X_train1)

X_train1 = scaler.transform(X_train1)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


########## BEST FOUND PARAMETERS #############
n1 = 75
n2 = 14
mid_act = 'relu' #useleakyrelu is enabled...
num_layers = 3
optimizer = 'adam'
activation = 'sigmoid' 
epo = 100 #10
bat = 44 #18
##############################################


#Build the model
useLeakyReLU = True  # as an "advanced" activation function, it must be added as its own layer not as a parameter on another layer

if useLeakyReLU == False:

    def classification_model(n1=n1, n2=n2, n3 =n2, mid_act = mid_act, num_layers = num_layers, optimizer = optimizer, activation = activation):
        model = Sequential()
        model.add(Dense(n1, input_dim=64, activation=mid_act))
        model.add(Dense(n2, activation=mid_act))
        for i in range(num_layers-2):
            model.add(Dense(n3, activation=mid_act))
        model.add(Dense(4, activation=activation))
        model.compile(optimizer= optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
else:
    def classification_model(n1=n1, n2=n2, n3 =n2, mid_act = mid_act, num_layers = num_layers, optimizer = optimizer, activation = activation):
        model = Sequential()
        model.add(Dense(n1, input_dim=64))
        model.add(LeakyReLU())
        model.add(Dense(n2))
        model.add(LeakyReLU())
        for i in range(num_layers-2):
            model.add(Dense(n3))
            model.add(LeakyReLU())
        model.add(Dense(4, activation=activation))
        model.compile(optimizer= optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

model = KerasClassifier(build_fn=classification_model, epochs=epo, batch_size=bat, verbose=0)


# CV Settings
useRandomCV = False;
useGridCV = False;

scorer = make_scorer(cohen_kappa_score)

# Using GridSearchCV to find optimum settings
time1 = time.time()

if useRandomCV:

    # specify parameters and distributions to sample from
    param_dist = {"n1": sp_randint(15, 80),
              "n2": sp_randint(10, 80),
              "n3": sp_randint(10, 80),              
              "epochs": sp_randint(30, 60),
              "batch_size": sp_randint(20, 100),
              "optimizer":['rmsprop', 'nadam', 'adagrad'],
              "activation": ['softmax', 'sigmoid', 'softplus']
              }

    
    n_iter_search = 100
    random_search = RandomizedSearchCV(model, param_distributions = param_dist, n_iter=n_iter_search, cv = 3, scoring=scorer, verbose=10)    
    random_search.fit(X_train, y_train)
    report(random_search.cv_results_)
    #scores = random_search.cv_results_['mean_test_score']
    
    
    
if useGridCV:

    # grid setup
    optimizers = ['adam']
    activations = ['softplus', 'sigmoid'] #['sigmoid', 'softmax', 'softplus']
    #inits = ['glorot_uniform', 'normal', 'uniform']
    epochs = [15] #range(10, 100, 20)
    batches = [33] #range(50, 500, 50)
    n1s = [73, 75, 77]
    n2s = [12, 14, 16]  
    n3s = [14, 16, 18]  #3 x 3 x 3 x 2 = 54
    param_grid = dict(nb_epoch = epochs, batch_size = batches, n1=n1s, n2=n2s, n3=n3s, optimizer=optimizers, activation = activations)
    grid = GridSearchCV(estimator = model, param_grid=param_grid, cv=3, verbose = 10, pre_dispatch = 4, scoring = scorer)
    grid_result = grid.fit(X_train, y_train)
    print("time elapsed: {}".format(time.time()-time1))
    
    best_model = grid.best_estimator_
    print(grid.best_score_, grid.best_params_)
    
    # What the test accuracy by class
    for motion_type in class_names:
        pred_score = best_model.score(X_val[y_val.motion_type==motion_type], y_val[y_val.motion_type==motion_type])
        print("{} accuracy = {p:8.4f}".format(motion_type, p=pred_score))
    
    history = best_model.fit(X_train, y_train)
    plt.plot(history.history['acc'])
    plt.plot(history.history['loss'])
    
    for motion_type in class_names:
        pred_score = best_model.score(X_val[y_val.motion_type==motion_type], y_val[y_val.motion_type==motion_type])
        print("{} accuracy = {p:8.4f}".format(motion_type, p=pred_score))
    
    # graphing mean training and mean test scores
    params = grid.cv_results_['params']
    n1 = [param['n1'] for param in params]
    n2 = [param['n2'] for param in params]
    plt.plot(n1, grid.cv_results_['mean_test_score'])
    plt.plot(n2, grid.cv_results_['mean_test_score'])
    plt.show()
    
    report(grid.cv_results_)

test_epochs = False
if test_epochs:
    test_parameter = 'Epochs'
    n_range = range(5, 200, 5) 
    
    scores= {}
    scores_list = []
    time_list = []
    
    for n in n_range:
        # Motions
        clear_session() #clear the keras session - omg so important!!!!
        t1 = time.time()
        print("looking at {} = {} on Motions Set".format(test_parameter, n))
        model = KerasClassifier(build_fn=classification_model, n2=n2, epochs=n, batch_size=bat, verbose=0)
        model.fit(X_train, y_train.values.ravel('C'))
        y_pred = model.predict(X_val)
        scores[n] = accuracy_score(y_val, y_pred)
        scores_list.append(scores[n])
        print("took {} seconds".format(time.time()-t1))
        time_list.append(time.time()-t1)
        
    # matplotlib is clunky in trying to plot bars side by side, BUT
    plot_lines1(scores_list, time_list, test_parameter, n_range, label='Motions', col='blue')    

plot_curves = False

if plot_curves:
    # Plot the learning curve of the best model found
    # use X_train1 and use learning_curve to do the cv's
    print(X_train1.shape, y_train1.shape)
    #from sklearn.model_selection import learning_curve
    title="learning curve for best model with extended epochs"
    
    model3 = KerasClassifier(build_fn=classification_model, optimizer='rmsprop', epochs=epo, batch_size=bat, verbose=0)
    model4 = KerasClassifier(build_fn=classification_model, optimizer='adam', epochs=epo, batch_size=bat, verbose=0)
    model5 = KerasClassifier(build_fn=classification_model, optimizer='adamax', epochs=epo, batch_size=bat, verbose=0)
    model6 = KerasClassifier(build_fn=classification_model, optimizer='adagrad', epochs=epo, batch_size=bat, verbose=0)

    start = time.time()
    history3 = model3.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0)
    t1 = time.time()
    history4 = model4.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0)
    t2 = time.time()
    history5 = model5.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0)
    t3 = time.time()
    history6 = model6.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0)
    t4 = time.time()
    
    print("rmsprop time: {} adam time: {}, adamax time:{}, adagrad time: {}".format(t1-start, t2-t1, t3-t2, t4-t3))
    
    x = np.arange(4)
    plt.bar(x, [t1-start, t2-t1, t3-t2, t4-t3], color='darkorchid')
    plt.ylabel('run time')
    plt.xticks(x, ('rmsprop', 'adam', 'adamax', 'adagrad'))
    plt.xlabel('num_layers')
    plt.show()
    
    labels = ['train-rmsprop', 'val-rmsprop', 'train-adam', 'val-adam', 'train-adamax', 'val-adamax', 'train-adagrad', 'val-adagrad']
    
    # summarize history for accuracy
    plt.plot(history3.history['acc'])
    plt.plot(history3.history['val_acc'])
    
    plt.plot(history4.history['acc'])
    plt.plot(history4.history['val_acc'])
    
    plt.plot(history5.history['acc'])
    plt.plot(history5.history['val_acc'])
    
    plt.plot(history6.history['acc'])
    plt.plot(history6.history['val_acc'])
    
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(labels, loc='lower right')
    plt.show()
    # summarize history for loss
    plt.plot(history3.history['loss'])
    plt.plot(history3.history['val_loss'])
    
    plt.plot(history4.history['loss'])
    plt.plot(history4.history['val_loss'])
    
    plt.plot(history5.history['loss'])
    plt.plot(history5.history['val_loss'])
    
    plt.plot(history6.history['loss'])
    plt.plot(history6.history['val_loss'])
    
    
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(0, 1.5)
    plt.legend(labels, loc='upper right')
    plt.show()


#Final Model

finalModel = True

if finalModel:
    best_model = KerasClassifier(build_fn=classification_model, epochs=epo, batch_size=bat, verbose=0)
    t_fit = time.time()
    best_model.fit(X_train1, y_train1, batch_size = bat, epochs = epo)  #train on the whole training set
    print("Fit time = {}".format(time.time()-t_fit))
    t_pred = time.time()
    y_pred = best_model.predict(X_test)
    print("Pred time = {}".format(time.time()-t_fit))
    for motion_type in class_names:
        pred_score = best_model.score(X_test[y_test.motion_type==motion_type], y_test[y_test.motion_type==motion_type])
        print("{} accuracy = {p:8.4f}".format(motion_type, p=pred_score))
    print("Cohen Kappa: {}".format(cohen_kappa_score(y_pred, y_test)))   
    print("Accuracy: {}".format(accuracy_score(y_pred, y_test)))
    print("F1 Score: {}".format(f1_score(y_pred, y_test, average = 'weighted')))
    print("Precision: {}".format(precision_score(y_pred, y_test, average='weighted')))
    print("Recall: {}".format(recall_score(y_pred, y_test, average='weighted')))


learning_curves = False
if learning_curves:
    estimator = KerasClassifier(build_fn=classification_model, epochs=100, batch_size=bat, verbose=0)
    #scorer = make_scorer(cohen_kappa_score)
    plot_learning_curves(estimator, X_train1, y_train1, title = "Neural Network - Motions Set - Post-Tuning Learning Curves", low_limit=0.6)
    

    
print("time elapsed: {}".format(time.time()-t0))




# References
# https://www.tensorflow.org/tutorials/keras/basic_classification
# https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# http://thedatascientist.com/performance-measures-cohens-kappa-statistic/
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py

