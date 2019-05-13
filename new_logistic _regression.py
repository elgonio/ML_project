import numpy as np 

class logistic_regressor():
    # reguralisation factor
    l = 0.0007

    step_size = 0.3
    #accuracy = 0

    def __init__(self,filename):
        self.data = np.genfromtxt(filename,delimiter=',')
        true_data = list()
        # convert the labels into the correct form
        for row in self.data:
            if row[3] - 2 < 0.1:
                row[0] = 0
                true_data.append(row)
                print("appending")
            elif row[3] - 3 < 0.1:
                row[0] = 1
                true_data.append(row)
                print("appending")


            
        self.data = np.array(true_data)
        print(self.data[:10])


        print(self.data.shape[1])
        # we add in a column of ones as theta 0
        b = np.ones((self.data.shape[0],self.data.shape[1]+1))
        b[:,1:] = self.data
        b[:,[0,1]] = b[:,[1,0]]
        self.data = b

        self.num_rows = self.data.shape[0]
        # we don't count the labels as part of the columns
        self.num_cols = self.data.shape[1]-1
        self.init_params()
        print("norm:",np.linalg.norm(self.params))

    def sigmoid(self,z):
        result = 1/(1+np.exp(-z))
        return result

    def sigmoid_deriv(self,z):
        result = self.sigmoid(z) * (1-self.sigmoid(z))
        return result

        #s = sigmoid(z)
        #return s * (1-s)

    def init_params(self):
        # generates an array of numbers in the range [0,1]
        self.params = np.random.rand(self.num_cols)
        # we want the params in the range [-1,1]
        self.params = self.params*2-1

        self.params *= 100

    def cost(self):
        result = 0
        for row in self.data:
            y = row[0]
            d_product = np.dot(self.params,row[1:])
            h_theta = self.sigmoid(d_product)
            #print("cost: d_product =", d_product, "h_theta = ", h_theta)
            # we get bugs if h_theta is exactly 1 or 0
            if h_theta == 1:
                h_theta = 1-10e-6
            elif h_theta == 0:
                h_theta = 10e-6

            #print("h_theta:", h_theta, "label:", y)
            
            temp = (y*np.log(h_theta) + (1-y)*np.log(1-h_theta)) 
            
            result = result - temp 

        result = result + self.l*np.sum(np.square(row)) 
        return result

    def MSE(self):
        count = 0
        total = 0
        for row in self.data:
            label = row[0]
            d_product = np.dot(self.params,row[1:])
            h_theta = self.sigmoid(d_product)
            if h_theta > 0:
                prediction = 1
            else:
                prediction = 0

            if prediction == label:
                count = count + 1

            total = total + (h_theta - label)**2

        #self.accuracy = count/self.num_rows

        total = (1/2*self.num_rows)*total

        return total



    def gradient_descent(self):
        
        for row in self.data:
            old_params = self.params+0
            y = row[0]

            h_theta = self.sigmoid(np.dot(old_params,row[1:]))

            new_params = np.zeros_like(self.params)
            new_params[0] = old_params[0] + self.step_size*self.sigmoid(h_theta) - row[0]
            
            for k in range(1,self.num_cols):
                self.params[k] = old_params[k] + self.step_size * ((h_theta-y)*row[k] - self.l*old_params[k])
            
    def train(self,tol):
        print("training...")
        diff = tol*10

        iteration_num = 0

        while diff > tol :
            old_params = self.params + 0

            self.gradient_descent()

            iteration_num = iteration_num + 1
            diff = np.linalg.norm(self.params-old_params)

            if iteration_num % 1 == 0:

                print("iteration:", iteration_num," cost:", self.cost(), "  diff:", diff,"  accuracy:", self.accuracy(), "    magnitude",np.linalg.norm(self.params) )
            
        print("____________________________________________________________________________________________________________")
        print("final iteration:", iteration_num," cost:", self.cost(), "  diff:", diff,"  accuracy:", self.accuracy(), "    magnitude",np.linalg.norm(self.params) )
        for row in self.data:
            label = row[0]
            prediction = self.sigmoid(np.dot(self.params,row[1:]))
            if prediction > 0.5:
                prediction = 1
            else:
                prediction = 0

            #print("prediction:", prediction, "actual", label)


    def accuracy(self):
        count = 0
        split_count = 0
        for row in self.data:

            label = row[0]

            if label == 1:
                split_count = split_count + 1

            prediction = self.sigmoid(np.dot(self.params,row[1:]))
            if prediction > 0.5:
                prediction = 1
            else:
                prediction = 0

            if prediction == label:
                count = count + 1

        print("split is :",split_count,self.data.shape[0]-split_count, split_count/self.data.shape[0])
        #self.accuracy = count/self.data.shape[0]
        return count/self.num_rows


lr = logistic_regressor("dota2ToyTest.csv")
lr.train(0.01)