# Imports
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.backend import clear_session
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
import time
#from utility1 import report
from scipy.stats import randint
from utility1 import load_data, plot_lines1, heatmap, report, plot_learning_curves
#from utility1 import plot_lines1


# Randomly divide into train and test sets
X_train1, X_test, y_train1, y_test, class_names = load_data('particles')

# Create a validation set
X_train, X_val, y_train, y_val = train_test_split( X_train1, y_train1, test_size = 0.3, random_state = 42)


# Scale
scaler = StandardScaler()
scaler.fit(X_train1)


X_val = scaler.transform(X_val)
X_train1 = scaler.transform(X_train1)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


########## BEST FOUND PARAMETERS #############
n1 = 47
n2 = 38
mid_act = 'relu'
num_layers = 1
optimizer = 'adamax'
activation = 'softmax'
epo = 100 #15
bat = 32 #18
##############################################



#Build the basic model using sparse_categorical_crossentropy as the loss function and pulling the best-identified parameters from above
input_dim = X_train.shape[1]
def classification_model(optimizer=optimizer, num_layers = num_layers, n1=n1, n2=n2, mid_act=mid_act, activation=activation): #optimizer='adam', num_layers = 4, n1=45, n2=30, mid_act = 'elu', activation='softplus'): #n1=47 and n2 = 38 was one of the higher performing, but moving to 20 for multilayer testing
    model = Sequential()
    model.add(Dense(n1, input_dim=input_dim, activation=mid_act))
    for n in range(num_layers):
        print("working on layer: {}".format(n))
        model.add(Dense(n2, activation=mid_act))
    model.add(Dense(4, activation=activation)) 
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=classification_model, epochs=epo, batch_size=bat, verbose=0)



#custom scorer
scorer = make_scorer(cohen_kappa_score)

grid_search = False
if grid_search:
    #let's grid search this thing
    time1 = time.time()
    optimizer_list = ['rmsprop'] #['Adam', 'RMSprop', 'Nadam']  #, 'Adagrad', 'Adadelta']
    activation = ['sigmoid'] #['softmax', 'softplus', 'sigmoid']
    param_grid = {'epochs':[42], 'batch_size':[50], 'optimizer':optimizer_list, 'n1':[32, 36, 40, 42, 46], 'n2':[10, 15, 20, 25, 30], 'activation': activation}  #'n1':range(37,40), 'n2':range(20,45,5)}
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, scoring = ['accuracy'], verbose = 10)
    grid_result = grid_search.fit(X_train, y_train)
    report(grid_result.cv_results_, n_top=10)
    print("GridSearchCV took %.2f seconds." % (time.time() - time1))


random_search= False  
if random_search:
    time1 = time.time()
    # specify parameters and distributions to sample from
    param_dist = {"optimizer": ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
                  "activation": ['softmax', 'softplus', 'softsign', 'tanh', 'sigmoid', 'hard_sigmoid'],
                  "n1": randint(1, 50),
                  "n2": randint(5, 50),
                  "epochs": randint(5, 60),
                  "batch_size":[32]
                  }
    # run randomized search
    n_iter_search = 200
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter_search, cv=3, verbose = 10) #scorer)
    
    start = time.time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), n_iter_search))
    report(random_search.cv_results_, n_top = 10)


t0 = time.time()
print("hi!")

hidden_neurons_exp = False
if hidden_neurons_exp:
    test_parameter = '# n2 neurons'
    n_range = range(5, 50, 1) 
    
    scores= {}
    scores_list = []
    time_list = []
    
    for n in n_range:
        # Motions
        clear_session() #clear the keras session - omg so important!!!!
        t1 = time.time()
        print("looking at {} = {} on Particles Set".format(test_parameter, n))
        model = KerasClassifier(build_fn=classification_model, n2=n, epochs=epo, batch_size=bat, verbose=0)
        model.fit(X_train, y_train.values.ravel('C'))
        y_pred = model.predict(X_val)
        scores[n] = accuracy_score(y_val, y_pred)
        scores_list.append(scores[n])
        print("took {} seconds".format(time.time()-t1))
        time_list.append(time.time()-t1)
        
    # matplotlib is clunky in trying to plot bars side by side, BUT
    plot_lines1(scores_list, time_list, test_parameter, n_range, label='Particles', col='green')

neuron_matrix = False
if neuron_matrix:
    test_parameter = 'neuron_size_matrix'
    n_range1 = range(4, 500, 30) 
    n_range2 = range(4, 500, 30) 
    
    scores = {}
    # Creates a list containing 5 lists, each of 8 items, all set to 0
    w, h = len(n_range1), len(n_range2)
    #scores = {}
    scores_list = np.zeros(shape=(w, h))
    time_list = np.zeros(shape=(w,h))
    
    
    for i in range(len(n_range1)):
        n_1 = n_range1[i]
        for j in range(len(n_range2)):
            clear_session() #clear the keras session - omg so important!!!!
            n_2 = n_range2[j]
            t1 = time.time()
            print("looking at {} = ({}, {}) on Particles Set".format(test_parameter, n_1, n_2))
            model = KerasClassifier(build_fn=classification_model, n1=n_1, n2=n_2, epochs=epo, batch_size=bat, verbose=0)
            model.fit(X_train, y_train.values.ravel('C'))
            y_pred = model.predict(X_val)
            #scores[n_1, n_2] = accuracy_score(y_val, y_pred)
            scores_list[i, j] =  accuracy_score(y_val, y_pred) #scores[n_1, n_2])
            print("took {} seconds".format(time.time()-t1))
            time_list[i, j] = time.time()-t1
        
        
    # matplotlib is clunky in trying to plot bars side by side, BUT
    #plot_lines1(scores_list, time_list, test_parameter, n_range1, label='Particles', col='green')
    
    plot_heatmaps = True
    if plot_heatmaps:
        #scores
        fig, ax = plt.subplots()

        im, cbar = heatmap(scores_list, n_range1, n_range2, ax=ax,
                           cmap="brg", xlabel='n1', ylabel='n2', cbarlabel="accuracy")
        #texts = annotate_heatmap(im, valfmt="{x:.2f}")
        
        fig.tight_layout()
        plt.show()
        
        #comptimes
        fig, ax = plt.subplots()

        im, cbar = heatmap(time_list, n_range1, n_range2, ax=ax,
                           cmap="jet", xlabel='n1', ylabel='n2', cbarlabel="computation time (sec)")
        #texts = annotate_heatmap(im, valfmt="{x:.2f}")
        
        fig.tight_layout()
        plt.show()
        
        
    
    
    
    #print("{} for best validation set accuracy on Motions: {}".format(test_parameter, max(scores, key=scores.get)))

num_epochs = 200
batch_size = 32


plot_curves = False
if plot_curves:
    # Plot the learning curve of the best model found
    # use X_train1 and use learning_curve to do the cv's
    print(X_train1.shape, y_train1.shape)
    title="learning curve for best model with extended epochs"
    
    model3 = KerasClassifier(build_fn=classification_model, epochs=num_epochs, n2=10, batch_size=batch_size, verbose=0)
    model4 = KerasClassifier(build_fn=classification_model, epochs=num_epochs, n2=20, batch_size=batch_size, verbose=0)
    model5 = KerasClassifier(build_fn=classification_model, epochs=num_epochs, n2=30, batch_size=batch_size, verbose=0)
    model6 = KerasClassifier(build_fn=classification_model, epochs=num_epochs, n2=40, batch_size=batch_size, verbose=0)

    start = time.time()
    history3 = model3.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0)
    t1 = time.time()
    history4 = model4.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0)
    t2 = time.time()
    history5 = model5.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0)
    t3 = time.time()
    history6 = model6.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0)
    t4 = time.time()
    
    print("model1 time: {} model2 time: {}, model3 time:{}, model4 time: {}".format(t1-start, t2-t1, t3-t2, t4-t3))
    
    x = np.arange(4)
    plt.bar(x, [t1-start, t2-t1, t3-t2, t4-t3], color='darkblue')
    plt.ylabel('run time')
    plt.xticks(x, ('10', '20', '30', '40'))
    plt.xlabel('dimension of hidden layer')
    plt.show()
    
    labels = ['train-10', 'val-10', 'train-20', 'val-20', 'train-30', 'val-30', 'train-40', 'val-40']
    
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
    plt.ylim(0.50,1.0)
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
    plt.ylim(0, 0.6)
    plt.legend(labels, loc='upper right')
    plt.show()


validationModel = False

if validationModel:
    best_model = KerasClassifier(build_fn=classification_model, batch_size=bat, epochs = 100, verbose=0)
    best_history = best_model.fit(X_train1, y_train1, validation_split = 0.2, verbose = 10)
    
    plt.plot(best_history.history['acc'])
    plt.plot(best_history.history['val_acc'])
    
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    #plt.ylim(0.50,1.0)
    plt.show()
    
    plt.plot(best_history.history['loss'])
    plt.plot(best_history.history['val_loss'])
    
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.ylim(0, 0.6)
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()
    
    #y_pred = best_model.predict(X_val)
    #for particle_type in class_names:
    #    pred_score = best_model.score(X_train[y_train.particle_type==particle_type], y_train[y_train.particle_type==particle_type])
    #    print("{} accuracy = {p:8.4f}".format(particle_type, p=pred_score))
    #print("Cohen Kappa: {}".format(cohen_kappa_score(y_pred, y_train)))   
    #print("Accuracy: {}".format(accuracy_score(y_pred, y_train)))
    #print("Balanced Accuracy: {}".format(balanced_accuracy_score(y_pred, y_train)))
    #print("F1 Score: {}".format(f1_score(y_pred, y_train, average = 'weighted')))
    #print("Precision: {}".format(precision_score(y_pred, y_train, average='weighted')))
    #print("Recall: {}".format(recall_score(y_pred, y_train, average='weighted')))



#Final Model

finalModel = True

if finalModel:
    best_model = KerasClassifier(build_fn=classification_model, batch_size=bat, verbose=0)
    t_fit = time.time()
    best_model.fit(X_train1, y_train1, batch_size = bat, epochs = epo)  #train on the whole training set
    print("Fit time = {}".format(time.time()-t_fit))
    t_pred = time.time()
    y_pred = best_model.predict(X_test)
    print("Pred time = {}".format(time.time()-t_fit))
    for particle_type in class_names:
        pred_score = best_model.score(X_test[y_test.id==particle_type], y_test[y_test.id==particle_type])
        print("{} accuracy = {p:8.4f}".format(particle_type, p=pred_score))
    print("Cohen Kappa: {}".format(cohen_kappa_score(y_pred, y_test)))   
    print("Accuracy: {}".format(accuracy_score(y_pred, y_test)))
    print("Balanced Accuracy: {}".format(balanced_accuracy_score(y_pred, y_test)))
    print("F1 Score: {}".format(f1_score(y_pred, y_test, average = 'weighted')))
    print("Precision: {}".format(precision_score(y_pred, y_test, average='weighted')))
    print("Recall: {}".format(recall_score(y_pred, y_test, average='weighted')))
  
    
learning_curves = False
if learning_curves:
    estimator = KerasClassifier(build_fn=classification_model, epochs=epo, batch_size=bat, verbose=0)
    #scorer = make_scorer(cohen_kappa_score)
    plot_learning_curves(estimator, X_train1, y_train1, title = "Neural Network - Particles Set - Post-Tuning Learning Curves", low_limit=0.6)
    
    
    
print("time elapsed: {}".format(time.time()-t0))


#References:
# borrowed heavily from
# https://www.tensorflow.org/tutorials/keras/basic_classification
# https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/