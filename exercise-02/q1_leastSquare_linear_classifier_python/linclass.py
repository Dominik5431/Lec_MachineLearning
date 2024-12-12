import numpy as np

def linclass(weight, bias, data):
    # Linear Classifier
    #
    # INPUT:
    # weight      : weights                (dim x 1)
    # bias        : bias term              (scalar)
    # data        : Input to be classified (num_samples x dim)
    #
    # OUTPUT:
    # class_pred       : Predicted class (+-1) values  (num_samples x 1)

    #####Insert your code here for subtask 1b#####
    # Perform linear classification i.e. class prediction
    def classify(x):
        return 1 if x>= 0 else -1
    
    weight_w_bias = np.zeros(len(weight)+1)
    weight_w_bias[:-1] = weight
    weight_w_bias[-1] = bias
    data_w_bias = np.ones((len(data[:,0]), len(data[0])+1))
    data_w_bias[:,:-1] = data
    class_pred = data_w_bias @ weight_w_bias 
    class_pred = np.array(list(map(classify, class_pred)))
    return class_pred


