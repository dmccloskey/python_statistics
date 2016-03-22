from .calculate_dependencies import *
from .calculate_base import calculate_base
class calculate_regression(calculate_base):
    def calculate_linearRegression(self,):
        '''
        EXAMPLE:
        http://www.analyticsvidhya.com/blog/2015/08/common-machine-learning-algorithms/
        #Import Library
        #Import other necessary libraries like pandas, np...
        from sklearn import linear_model
        #Load Train and Test datasets
        #Identify feature and response variable(s) and values must be numeric and numpy arrays
        x_train=input_variables_values_training_datasets
        y_train=target_variables_values_training_datasets
        x_test=input_variables_values_test_datasets
        # Create linear regression object
        linear = linear_model.LinearRegression()
        # Train the model using the training sets and check score
        linear.fit(x_train, y_train)
        linear.score(x_train, y_train)
        #Equation coefficient and Intercept
        print('Coefficient: \n', linear.coef_)
        print('Intercept: \n', linear.intercept_)
        #Predict Output
        predicted= linear.predict(x_test)
        '''
        pass;
    def calculate_RidgeClassifier(self,):
        '''
        EXAMPLE:
        linear_model.RidgeClassifier()
        '''
        pass;
    def calculate_logisticRegression(self,):
        '''
                
        EXAMPLE:
        http://www.analyticsvidhya.com/blog/2015/08/common-machine-learning-algorithms/
        #Import Library
        from sklearn.linear_model import LogisticRegression
        #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
        # Create logistic regression object
        model = LogisticRegression()
        # Train the model using the training sets and check score
        model.fit(X, y)
        model.score(X, y)
        #Equation coefficient and Intercept
        print('Coefficient: \n', model.coef_)
        print('Intercept: \n', model.intercept_)
        #Predict Output
        predicted= model.predict(x_test)
        '''
        pass;