from .calculate_dependencies import *
from .calculate_base import calculate_base
class calculate_classification(calculate_base):
    
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