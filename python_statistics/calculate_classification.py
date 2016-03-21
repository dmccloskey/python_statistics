from .calculate_dependencies import *
from .calculate_base import calculate_base
class calculate_classification(calculate_base):

    def calculate_svm(self):
        '''SVM
        http://scikit-learn.org/stable/modules/svm.html
        
        EXAMPLE:
        http://www.analyticsvidhya.com/blog/2015/08/common-machine-learning-algorithms/
        #Import Library
        from sklearn import svm
        #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
        # Create SVM classification object 
        model = svm.svc() # there is various option associated with it, this is simple for classification. You can refer link, for mo# re detail.
        # Train the model using the training sets and check score
        model.fit(X, y)
        model.score(X, y)
        #Predict Output
        predicted= model.predict(x_test)
        '''

    def calculate_randomForest(self):
        '''Random Forest
        http://scikit-learn.org/stable/modules/svm.html

        EXAMPLE:
        http://www.analyticsvidhya.com/blog/2015/08/common-machine-learning-algorithms/
        #Import Library
        from sklearn.ensemble import RandomForestClassifier
        #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
        # Create Random Forest object
        model= RandomForestClassifier()
        # Train the model using the training sets and check score
        model.fit(X, y)
        #Predict Output
        predicted= model.predict(x_test)
        '''

    def calculate_decisionTree(self):
        '''Decision Tree
        sklearn.tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, presort=False)
        
        EXAMPLE:
        http://www.analyticsvidhya.com/blog/2015/08/common-machine-learning-algorithms/
        #Import Library
        #Import other necessary libraries like pandas, numpy...
        from sklearn import tree
        #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
        # Create tree object 
        model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
        # model = tree.DecisionTreeRegressor() for regression
        # Train the model using the training sets and check score
        model.fit(X, y)
        model.score(X, y)
        #Predict Output
        predicted= model.predict(x_test)
        '''

    def calculate_naiveBayes(self):
        '''
       
        EXAMPLE:
        http://www.analyticsvidhya.com/blog/2015/08/common-machine-learning-algorithms/
        #Import Library
        from sklearn.naive_bayes import GaussianNB
        #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
        # Create SVM classification object model = GaussianNB() # there is other distribution for multinomial classes like Bernoulli Naive Bayes, Refer link
        # Train the model using the training sets and check score
        model.fit(X, y)
        #Predict Output
        predicted= model.predict(x_test)
        '''
        pass;

    def calculate_KNN(self):
        '''
       
        EXAMPLE:
        http://www.analyticsvidhya.com/blog/2015/08/common-machine-learning-algorithms/
        #Import Library
        from sklearn.neighbors import KNeighborsClassifier
        #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
        # Create KNeighbors classifier object model 
        KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5
        # Train the model using the training sets and check score
        model.fit(X, y)
        #Predict Output
        predicted= model.predict(x_test)
        '''
        pass;