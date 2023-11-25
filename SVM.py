import numpy as np


class SVM:
    def __init__(self,lr=.001,lambda_param=.01,n_iters=1000) :

        self.lr=lr
        self.lambda_param=lambda_param
        self.n_iters=n_iters
        self.W=None
        self.b=None
    

    def fit(self,X,y):
        n_samples,n_features=X.shape
        
        
        y_=np.where(y<=0,-1,1)
        
        

        
        self.W=np.zeros(n_features)
        self.b=0


        for _ in range(self.n_iters):
            for ind,x_i in enumerate(X):

                equ=y_[ind]*np.dot(x_i,self.W)-self.b
                if(equ>=1):
                    self.W-=self.lr*(2*self.lambda_param*self.W)
                    self.b-=self.lr *y_[ind]

                else:
                    self.W-=self.lr*((2*self.lambda_param*self.W)-(np.dot(x_i,y_[ind])))
                    self.b-=self.lr *y_[ind]
                
    
    def predict(self,X):
        y_pred=np.dot(X,self.W) -self.b
        return np.sign(y_pred)

                                     
                                     