#self taught tutorial of machine learning
#src: https://www.w3schools.com/python/python_ml_grid_search.asp
import matplotlib.pyplot as plt
import numpy
from scipy import stats
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.model_selection import LeavePOut, cross_val_score
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#Using linear regression to predict future values
def regressionPrediction(age, slope, intercept):
  return slope * age + intercept
def carPrediction():
    car_age = [5,7,8,7,2,17,2,9,4,11,12,9,6]
    car_speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
    slope, intercept, r, p, std_err = stats.linregress(car_age, car_speed)
    age = int(input("What is the age of the car in years? "))
    print("Predicted speed: ",regressionPrediction(age, slope, intercept))

def polynomialRegression():
    time = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
    car_speed = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]
    mymodel = numpy.poly1d(numpy.polyfit(time, car_speed, 3))
    #numpy makes polynomial model and can use to predict values
    speed = mymodel(17)
    print("Speed at 17:00:",speed)
    myline = numpy.linspace(1, 22, 100)
    #start position 1 and end 22
    plt.scatter(time, car_speed)
    plt.plot(myline, mymodel(myline))
    plt.show()

def badFit():
    #can find if data is a good fit by finding the R^2 value and a low value indicates a bad fit
    x = [89, 43, 36, 36, 95, 10, 66, 34, 38, 20, 26, 29, 48, 64, 6, 5, 36, 66, 72, 40]
    y = [21, 46, 3, 35, 67, 95, 53, 72, 58, 10, 26, 34, 90, 33, 38, 20, 56, 2, 47, 15]
    mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
    print("Bad fit:",r2_score(y, mymodel(x)))
    x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
    y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]
    mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
    print("Good fit:",r2_score(y, mymodel(x)))
    #closer to 1 indicates better fit

def pandasMultipleRegression():
    #using multiple variables to predict one other variable (volume and weight to predict co2)
    df = pd.read_csv("data.csv")
    X = df[["Volume","Weight"]]
    y = df["CO2"]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    #predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
    predictedCO2 = regr.predict([[2300, 1300]])
    print(predictedCO2)
    #can get regression coefficient which shows the difference if other variables are in/decreased
    print(regr.coef_)

def scaleData():
    #rather than compare different units eg. weight 750 and volume 1 data is scaled
    #z = (x - u) / s with z as new value, x original, u mean, and s std and done auto in sklearn StandardScaler()
    #need from sklearn.preprocessing import StandardScaler
    df = pd.read_csv("data.csv")
    #X = df[['Weight', 'Volume']]
    #scaledX = scale.fit_transform(X)
    #print(scaledX)

def trainingTesting():
    #train/test needed to measure accuracy of model as split into training and testing sets
    numpy.random.seed(2)
    x = numpy.random.normal(3, 1, 100)
    y = numpy.random.normal(150, 40, 100) / x
    #create random polynomial dataset
    train_x = x[:80]
    train_y = y[:80]
    test_x = x[80:]
    test_y = y[80:]
    #splits data in 80% training and 20% testing
    mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
    #find r2 to see if good fit for prediction
    r2 = r2_score(train_y, mymodel(train_x))
    print(r2)
    #predict a customers spend after 5 minutes in shop (x time, y spend)
    print(mymodel(5))
    #create a plot with prediction line
    myline = numpy.linspace(0, 6, 100)
    plt.scatter(train_x, train_y)
    plt.plot(myline, mymodel(myline))
    plt.show()

def decisionTree():
    #to predict decisions a flow chart is needed and can predict outcomes based on previous decisions
    #strings must be numerical dictionaries have been used for this
    df = pd.read_csv("data2.csv")
    d = {'UK': 0, 'USA': 1, 'N': 2}
    df['Nationality'] = df['Nationality'].map(d)
    d = {'YES': 1, 'NO': 0}
    df['Go'] = df['Go'].map(d)
    #y is the ultimate decision wanted and features are factors
    features = ['Age', 'Experience', 'Rank', 'Nationality']
    X = df[features]
    y = df['Go']
    dtree = DecisionTreeClassifier()
    #used .values because warning using valid feature names
    dtree = dtree.fit(X.values, y)
    tree.plot_tree(dtree, feature_names=features)
    #every comedian with a rank of or lower follows true, gini refers to quality of split between 0.0 and 0.5
    #0.5 indicates a perfect split and 0.0 indicates all samples got same result
    #samples refer to number of comedians left at this stage
    #values refer to the amount that get no and go
    #can use tree to predict eg. should I go see a show starring a 40 years old
    #American comedian, with 10 years of experience, and a comedy ranking of 7?
    print("40Y USA 10Y R7:",dtree.predict([[40, 10, 7, 1]]))
    #what if 20 and comedy ranking 4?
    print("20Y USA 10Y R4:",dtree.predict([[20, 10, 4, 1]]))
    plt.show()

def practiceDecisionTree():
    #made a practice dataset to test own decision tree using age and exercise to predict fitness
    df = pd.read_csv("data3.csv")
    d = {"YES": 1, "NO": 0}
    df["Fit"] = df["Fit"].map(d)
    features = ["Age","Exercise"]
    X = df[features]
    y = df["Fit"]
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X.values, y)
    tree.plot_tree(dtree, feature_names=features)
    #decision tree estimate: if does more than 6.5 exercise and younger than 65 then fit
    #therefore is 70 year old who does 5 exercise fit?
    print("70Y 5E:", dtree.predict([[70, 5]]))
    #is 40 year old who does 7 exercise fit?
    print("40Y 7E:", dtree.predict([[40, 7]]))
    plt.show()

def confusionMatrix():
    #table that represents classes the predictions belong to
    actual = numpy.random.binomial(1, .9, size=1000)
    predicted = numpy.random.binomial(1, .9, size=1000)
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
    cm_display.plot()
    plt.show()
    #true-ve top left false-ve bottom left true+ve bottom right false+ve top right
    #from this matric you can evaluate the model through
    #accuracy - how often model correct
    Accuracy = metrics.accuracy_score(actual, predicted)
    print(Accuracy)
    #precision - of positives predicted what percentage is true positive
    Precision = metrics.precision_score(actual, predicted)
    print(Precision)
    #sensitivity (recall) - how good is the model at +ve predictions (compares true +ve w/ false -ve)
    Sensitivity_recall = metrics.recall_score(actual, predicted)
    print(Sensitivity_recall)
    #specificity - how good at predicting -ve results (opposite of recall)
    Specificity = metrics.recall_score(actual, predicted, pos_label=0)
    print(Specificity)
    #f-score - considers both false -ve and +ve for imbalanced datasets
    #"harmonic mean" of precision and sensitivity
    F1_score = metrics.f1_score(actual, predicted)
    print(F1_score)

def hierarchicalClustering():
    #unsupervised learning method for clustering data points
    #bottom-up process forming clusters between variables with shortest distance until one large cluster forms
    #euclidean distance and the Ward linkage method, which attempts to minimize the variance between clusters
    x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
    y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
    # visualising linkage through dendrogram
    # turn data into set of points eg. [(2,3),(3,5)...]
    #data = list(zip(x, y))
    #compute linkage between all points using ward and euclidean to minimize variance between clusters
    #linkage_data = linkage(data, method='ward', metric='euclidean')
    #dendrogram(linkage_data)
    #plot shows hierarchy of clusters from the bottom, individual points, to the top, a single cluster
    #plt.show()
    #same thing completed with sklearn and visualised on a 2d plot
    data = list(zip(x, y))
    hierarchical_cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
    #The .fit_predict method can be called on our data to compute the
    #clusters using the defined parameters across our chosen number of clusters.
    labels = hierarchical_cluster.fit_predict(data)
    plt.scatter(x, y, c=labels)
    plt.show()

def logisticRegression():
    #solves classification problems through predicting categorical outcomes unlike linear which is continuous outcome
    #binomial - 2, multinomial - >2
    #example dataset:
    # X represents the size of a tumor in centimeters.
    X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)
    # Note: X has to be reshaped into a column from a row for the LogisticRegression() function to work.
    # y represents whether or not the tumor is cancerous (0 for "No", 1 for "Yes").
    y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    #sklearn has logistic regression function
    logr = linear_model.LogisticRegression()
    logr.fit(X, y)
    # predict if tumor is cancerous where the size is 3.46mm:
    predicted = logr.predict(numpy.array([3.46]).reshape(-1, 1))
    print("3.46cm cancerous:",predicted)
    #coefficient is the expected change in log-odds of outcome per unit change in X (odds)
    log_odds = logr.coef_
    #calculated exponential
    odds = numpy.exp(log_odds)
    print("Odds increase x:",odds)
    #this means that if a tumour size increases by 1mm the odds of it being cancerous increase by 4x
    #by using the coefficient and intercept values the probability of each tumour being cancerous can be found
    #the following is the formula and to find the % you must divide the exp by 1+exp
    log_odds = logr.coef_ * X + logr.intercept_
    odds = numpy.exp(log_odds)
    probability = odds / (1 + odds)
    print("Probabilities:\n",probability)

def gridSearch():
    #machine learning contains parameters to adjust how the model learns
    #in logistic regression the parameter is C that controls regularization which affects complexity
    #>C represent training data resembling real world data and places a greater weight, lower do the opposite
    #best value for C is dependent on the data used to train the model
    #one method is to try out different values and use one that gets the best score this is GRID SEARCH
    iris = datasets.load_iris()
    X = iris['data']
    y = iris['target']
    #logit is essentially exp of logodds
    logit = LogisticRegression(max_iter=10000)
    #print(logit.fit(X, y))
    logit.fit(X, y)
    #sigmoidal function y-probability x-logit ranging from ~-5,0%:5,100%
    print(logit.score(X, y))
    #therefore 97% fit
    #with practice and domain knowledge values to test for C will be known
    #however default is 1 therefore test around
    C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    scores = []
    for choice in C:
        logit.set_params(C=choice)
        logit.fit(X, y)
        scores.append(logit.score(X, y))
    print(scores)
    #it appears that ~1.75 increases accuracy and beyond does not change much (sigmoidal)

def categoricalData():
    #machine learning models often only accept numeric data so categorical needs to be transformed
    #can use one hot encoding by having multiple columns represent each category with a 1 or 0
    #this is a function in pandas using get_dummies()
    cars = pd.read_csv('data.csv')
    ohe_cars = pd.get_dummies(cars[['Car']])
    #print(ohe_cars.to_string())
    #select X independent variables and add dummyvariables columnwise
    #concatenated data together (extra ohe cars)
    X = pd.concat([cars[['Volume', 'Weight']], ohe_cars], axis=1)
    y = cars['CO2']
    regr = linear_model.LinearRegression()
    regr.fit(X.values, y)
    ##predict the CO2 emission of a Volvo where the weight is 2300kg, and the volume is 1300cm3:
    predictedCO2 = regr.predict([[2300, 1300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    print(predictedCO2)
    #can use one column less
    #eg red and blue instead of r 0 b 1 could be r 0 not red therefore blue
    #can use drop_first=True in get_dummies()
    #colors = pd.DataFrame({'color': ['blue', 'red']})
    #dummies = pd.get_dummies(colors, drop_first=True)

def kMeans():
    #unsupervised method to cluster data points by minimizing variance
    #select k and cluster no. and computes centroid of each cluster to closest centroids
    #elbow method graphs inertia and visualise when the distances decrease
    #good to estimate best value for k based on data
    x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
    y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
    data = list(zip(x, y))
    inertias = []
    #must find best value for k, has 10 data points so 10 max clusters
    #choose k value where elbow becomes more linear which is k=2
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    #plt.plot(range(1, 11), inertias, marker='o')
    #plt.title('Elbow method')
    #plt.xlabel('Number of clusters')
    #plt.ylabel('Inertia')
    #plt.show()
    #shows k=2 is a good value
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)
    plt.scatter(x, y, c=kmeans.labels_)
    plt.show()

def bagging():
    #bagging/boostrap aggregation
    #decision trees can be prone to overfitting training sets leading to wrong predictions on new data
    #bagging attempts to resolve overfitting for classification or regression problems
    #does this by taking random subsets of data and fits a classifier or regressor to each
    #predictions are aggregated through majority vote for class or average for regression to increase accuracy
    #as_frame=True to ensure feature names are not lost when loading data
    data = datasets.load_wine(as_frame=True)
    X = data.data
    y = data.target
    #need to split data into test and train sets
    #randomstate just generates random numbers
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)
    #base classifier fit to training data
    #this is the default decision tree method
    dtree = DecisionTreeClassifier(random_state=22)
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    print("Train data accuracy:", accuracy_score(y_true=y_train, y_pred=dtree.predict(X_train)))
    print("Test data accuracy:", accuracy_score(y_true=y_test, y_pred=y_pred))
    #the following is bootstrap aggregation to compare the accuracy
    #n_estimators is number of base classifiers the model aggregates
    #in this model it is relatively low and hyperparameter tuning is often done with grid search
    estimator_range = [2, 4, 6, 8, 10, 12, 14, 16]
    models = []
    scores = []
    for n_estimators in estimator_range:
        #create bagging classifier
        clf = BaggingClassifier(n_estimators=n_estimators, random_state=22)
        #fit the model
        clf.fit(X_train, y_train)
        #append the model and score to their respective list
        models.append(clf)
        scores.append(accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))
    #generate the plot of scores against number of estimators
    plt.figure(figsize=(9, 6))
    plt.plot(estimator_range, scores)
    #adjust labels and font (to make visable)
    plt.xlabel("n_estimators", fontsize=18)
    plt.ylabel("score", fontsize=18)
    plt.tick_params(labelsize=16)
    #visualize plot
    plt.show()
    #therefore by using n_estimates as 12 there is a 13.3% increase in accuracy of the model using bagging
    #can use out of bag observations to evaluate a model however can overestimate error in binary classification
    #should be in compliment of other metrics

def baggingKnownNEstimators():
    #can visualise decision tree used to vote for final predictions within model
    data = datasets.load_wine(as_frame=True)
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)
    clf = BaggingClassifier(n_estimators=12, oob_score=True, random_state=22)
    clf.fit(X_train, y_train)
    plt.figure(figsize=(30, 20))
    plot_tree(clf.estimators_[0], feature_names=X.columns)
    plt.show()

def crossValidation():
    #hyperparameter tuning can lead to better performance however can lead to information leakage leading to
    #worse performance on unseen data, can correct for this with cross validation
    #there are many methods of CV the following is
    #k-fold:
    #training data is split into k smaller sets to validate model and trained on k-1 folds of training set
    #remaining set is used as a validation set to evaluate model
    X, y = datasets.load_iris(return_X_y=True)
    clf = DecisionTreeClassifier(random_state=42)
    k_folds = KFold(n_splits=5)
    scores = cross_val_score(clf, X, y, cv=k_folds)
    print("Cross Validation Scores: ", scores)
    print("Average CV Score: ", scores.mean())
    print("Number of CV Scores used in Average: ", len(scores))
    #stratified k-fold:
    #classes are imbalanced so need to stratify target classes meaning both sets have an equal proportion of all classes
    X, y = datasets.load_iris(return_X_y=True)
    clf = DecisionTreeClassifier(random_state=42)
    sk_folds = StratifiedKFold(n_splits=5)
    scores = cross_val_score(clf, X, y, cv=sk_folds)
    print("Cross Validation Scores: ", scores)
    print("Average CV Score: ", scores.mean())
    print("Number of CV Scores used in Average: ", len(scores))
    #leave-one-out (LOO):
    #utilize 1 observation to validate and n-1 observations to train. this is an exhaustive technique
    X, y = datasets.load_iris(return_X_y=True)
    clf = DecisionTreeClassifier(random_state=42)
    loo = LeaveOneOut()
    scores = cross_val_score(clf, X, y, cv=loo)
    print("Cross Validation Scores: ", scores)
    print("Average CV Score: ", scores.mean())
    print("Number of CV Scores used in Average: ", len(scores))
    #leave-p-out (LPO):
    #same as before but can select p value
    X, y = datasets.load_iris(return_X_y=True)
    clf = DecisionTreeClassifier(random_state=42)
    lpo = LeavePOut(p=2)
    scores = cross_val_score(clf, X, y, cv=lpo)
    print("Cross Validation Scores: ", scores)
    print("Average CV Score: ", scores.mean())
    print("Number of CV Scores used in Average: ", len(scores))
    #shuffle split:
    #leaves out a percentage of data not to be used in train or validation
    #must decide train and test size and split no.
    X, y = datasets.load_iris(return_X_y=True)
    clf = DecisionTreeClassifier(random_state=42)
    ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits=5)
    scores = cross_val_score(clf, X, y, cv=ss)
    print("Cross Validation Scores: ", scores)
    print("Average CV Score: ", scores.mean())
    print("Number of CV Scores used in Average: ", len(scores))

def AUC_ROC_Curve():
    #common evaluation metric is accuracy
    #can use AUC - ROC (area under the receiver operating characteristic) curve
    #this plots the true positive rate versus false positive rate at different class thresholds
    #eg imbalanced data can be majority right by predicting majority class
    n = 10000
    ratio = .95
    n_0 = int((1 - ratio) * n)
    n_1 = int(ratio * n)
    y = numpy.array([0] * n_0 + [1] * n_1)
    # below are the probabilities obtained from a hypothetical model that always predicts the majority class
    # probability of predicting class 1 is going to be 100%
    y_proba = numpy.array([1] * n)
    y_pred = y_proba > .5
    print(f'accuracy score: {accuracy_score(y, y_pred)}')
    cf_mat = confusion_matrix(y, y_pred)
    print('Confusion matrix')
    print(cf_mat)
    print(f'class 0 accuracy: {cf_mat[0][0] / n_0}')
    print(f'class 1 accuracy: {cf_mat[1][1] / n_1}')
    #this model whilst can accurately predict class 1 100% of the time it can never predict class 0
    #therefore at the expense of accuracy it is better to have a model that can seperate the classes
    # below are the probabilities obtained from a hypothetical model that doesn't always predict the mode
    y_proba_2 = numpy.array(
        numpy.random.uniform(0, .7, n_0).tolist() +
        numpy.random.uniform(.3, 1, n_1).tolist()
    )
    y_pred_2 = y_proba_2 > .5
    print(f'accuracy score: {accuracy_score(y, y_pred_2)}')
    cf_mat = confusion_matrix(y, y_pred_2)
    print('Confusion matrix')
    print(cf_mat)
    print(f'class 0 accuracy: {cf_mat[0][0] / n_0}')
    print(f'class 1 accuracy: {cf_mat[1][1] / n_1}')
    #this accuracy is more balanced even if class 1 accuracy has dropped
    #using just accuracy as a evaluation metric the first model would be rated higher
    #however this tells us nothing about the data therefore use AUC
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    print(f'model 1 AUC score: {roc_auc_score(y, y_pred)}')
    fpr, tpr, thresholds = roc_curve(y, y_pred_2)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    print(f'model 2 AUC score: {roc_auc_score(y, y_pred_2)}')
    #model 2 has a higher AUC score closer to 1 therefore model has the ability to seperate the two classes
    #AUC utilizes probabilities of class predictions, you can be more confident in a model with higher AUC
    #even if they have similar accuracies

def KNN():
    #K-nearest neighbour
    #simple unsupervised ML algorithm which can be used for classification or regression
    #freq used in missing value imputation
    #k is the number of nearest neighbours to use
    #in classification a majority vote is used to determine a class
    #larger k are more robust to outliers and have more stable decision boundaries
    x = [4, 5, 10, 4, 3, 11, 14, 8, 10, 12]
    y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
    classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]
    plt.scatter(x, y, c=classes)
    plt.show()
    data = list(zip(x, y))
    #using k=1 a new datapoint is classified
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(data, classes)
    new_x = 8
    new_y = 21
    new_point = [(new_x, new_y)]
    prediction = knn.predict(new_point)
    plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
    plt.text(x=new_x - 1.7, y=new_y - 0.7, s=f"new point, class: {prediction[0]}")
    plt.show()
    #now again with a higher k value
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(data, classes)
    prediction = knn.predict(new_point)
    plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
    plt.text(x=new_x - 1.7, y=new_y - 0.7, s=f"new point, class: {prediction[0]}")
    plt.show()
    #using a higher number of neighbours the classification of the point changes