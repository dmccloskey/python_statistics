from .calculate_dependencies import *
from .calculate_base import calculate_base
from .calculate_statisticsDescriptive import calculate_statisticsDescriptive
from .calculate_statisticsUnivariate import calculate_statisticsUnivariate

'''TODO: https://lmfit.github.io/lmfit-py/intro.html'''

class calculate_curveFitting(calculate_base):
    # linear regression
    def calculate_regressionParameters(self,concentrations_I,ratios_I,dilution_factors_I,fit_I,weighting_I,use_area_I):
        '''calculate regression parameters for a given component
        NOTE: intended to be used in a loop
        INPUT:
            concentrations_I
            ratios_I
            dilution_factors_I
            fit_I
            weighting_I
            use_area_I
        OUTPUT:
            slope
            intercept
            correlation
            lloq
            uloq
            points

        note need to make complimentary method to query concentrations, ratios, and dilution factors
        for each component prior to calling this function

        TODO:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        '''

        pass;

    def calculate_curveFit(self,func_I):
        '''Fit data to a curve
        INPUT:
        func_I = function, 
        EXAMPLE:
        import numpy as np
        from scipy.optimize import curve_fit

        xdata = np.array([-2,-1.64,-1.33,-0.7,0,0.45,1.2,1.64,2.32,2.9])
        ydata = np.array([0.699369,0.700462,0.695354,1.03905,1.97389,2.41143,1.91091,0.919576,-0.730975,-1.42001])

        def func(x, p1,p2):
          return p1*np.cos(p2*x) + p2*np.sin(p1*x)

        #generate the fit
        popt, pcov = curve_fit(func, xdata, ydata,p0=(1.0,0.2))        
        
        #calculate the SSR
        p1 = popt[0]
        p2 = popt[1]
        residuals = ydata - func(xdata,p1,p2)
        fres = sum(residuals**2)
        '''
        pass;

    def fitFunction_sigmoidalMetabolitesEvolution(self,p1,p2,p3,p4,g):
        '''general sigmoidal curve function
        y(g) = p1 + (p2-p1)/(1+exp(p3-g)*p4)
        Where
        p1 = minimum value
        p2 = difference between the min and max value
        p3 = half maximal effective concentration (EC50)
        p4 = slope
        g = # of generations
        '''
        yg = p1 + (p2-p1)/(1+np.exp(p3-g)*p4);
        return yg;

    def makeParametersFromData_sigmoidalMetabolitesEvolution(self,data_I):
        '''general sigmoidal curve function
        INPUT:
        data_I = [] of float, data array
        OUTPUT:
        p1 = minimum value
        p2 = difference between the min and max value
        '''
        p1 = min(data_I);
        p2 = max(data_I)-min(data_I);

        return p1,p2;

    def fitFunction_logistic(self,x,xo,L,k):
        '''Logistic function
        INPUT:
        x = input data
        L = curves maximum value
        k = steepness of the curve
        xo = the x-value of the sigmoid's midpoint
        '''
        fx = L/(1+np.exp(-k*(x-xo)));
        return fx;

    def fitFunction_logisticGeneralized(self,x,xo,p1,p2,k):
        '''Logistic function
        INPUT:
        x = input data
        p1 = minimum value
        p2 = difference between the min and max value
        k = steepness of the curve
        xo = the x-value of the sigmoid's midpoint
        '''
        fx = p1 + (p2-p1)/(1+np.exp(-k*(x-xo)));
        return fx;

    def makeParametersFromData_logisticGeneralized(self,data_I,time_I,index_I):
        '''Logistic function
        INPUT:
        data_I = [] of float, data array
        time_I = [] of float, time-course, generation-course, etc., array
        index_I = int, index of data in data_I,time_I
        OUTPUT:
        p1 = minimum value
        p2 = difference between the min and max value
        k = steepness of the curve
        xo = the x-value of the sigmoid's midpoint
        '''
        p1 = min(data_I);
        p2 = max(data_I)-min(data_I);
        return p1,p2;

    def fitFunction_gaussian(self,x,a,b,c):
        '''Gaussian function
        INPUT:
        x = input data
        a = hight of the curves peak
        b = is the position of the center of the peak
        c = width of the bell
        '''
        fx = a*np.exp(-(x-b)^2/(2*c^2));
        return fx;

    def fitFunction_lognormal(self,x,mu,sigma):
        '''log-normal function
        INPUT:
        
        '''
        a = 1/(x*sigma*np.sqrt(2*np.pi));
        xb = np.log(x)-mu;
        fx = a*np.exp(-xb^2/(2*sigma^2));
        return fx;

    def fitFunction_polynomial(self,x):
        '''Polynomial function
        fx = (x+a,)^min + (x+a,-1)^-1 + x0 + (x+a1)^1 + (x+a2)^2 + ... + (x+ai)^i
        INPUT:
        xo = starting value
        a,i = intercepts for degree i

        '''

    def calculate_adjustedR2(self,r2_I,N_I,p_I):
        '''Calculate the adjusted r2 value
        INPUT:
        r2_I = sample R-square
        p = number of predictors
        N = total sample size
        OUTPUT:
        r2adj_O = adjusted R-squared
        '''
        r2adj_O = 1-(1-r2_I)*(N_I-1)/(N_I-p_I-1);

    def calculate_R2(self,data_I,fit_I):
        '''Calculate the r2 from the data and fit
        INPUT:
        data_I = array of data
        fit_I = array of fitted data
        OUTPUT:
        rse_O = residual standard error
        r2_O = r2 value
        rho_O = pearson correlation coefficient
        pval_O = p-value of the correlation
        '''
        #calculate SSREG,SSTOT,r2
        yhat = fit_I;
        n = len(data_I);
        ybar = np.sum(data_I)/n;
        ssreg = np.sum((yhat-ybar)**2)
        sstot = np.sum((y - ybar)**2)        
        r2_O = ssreg/sstot; #check: 1-ssreg/sstot
        #calculate RSE
        rse_O = np.sqrt(1/(n-2)*ssreg);
        #calculate the correlation coefficient and p-value
        rho_O, pval_O = scipy.stats.pearsonr(yhat,ybar);

        if rho_O**2!=r2_O:
            print('pearson_r**2 and ssreg/sstot yield different values');
        return rse_O,r2_O,rho_O,pval_O;

    def calculate_tStatParameter(self,par_val_I,par_se_I,par_n_I):
        '''calculate the t-stat for the parameter
        INPUT:
        par_val_I = parameter value
        par_se_I = parmater std error
        par_n_I = number of parameters
        OUTPUT:
        tstat_O = two-sided t-test statistic
        pval_O = p-value
        '''
        from scipy.special import stdtr
        tstat_O = (par_val_I-0.0)/par_se_I;
        dof = par_n_I-2;
        pval_O = 2*stdtr(data_1_dof, -np.abs(tf));
        return tstat_O,pval_O;
