import numpy as np
from collections import Counter
from math import log2

class NaiveBayes():
    def __init__(self,smooth=0.5):
        self.prior = {}
        self.condition = {}
        self.smooth = max(0,smooth)#平滑参数，输入负数或者0等于不平滑

    def trainer(self,data,):
        Y = data[:,-1]
        priorCounter = Counter(Y)
        freq = {a[0]:a[1] for a in priorCounter.most_common()}
        freq[0] = freq[0]+self.smooth if 0 in freq else self.smooth
        freq[1] = freq[1]+self.smooth if 1 in freq else self.smooth
        print("begin to compute prior probability")
        #输出话先验概率
        total = len(Y)+self.smooth+self.smooth#总共两类，加两次平滑值
        self.prior = {elem : log2(freq[elem])-log2(total) for elem in freq}
        #初始化条件概率
        for state in freq:
            self.condition[state] = {}
            for i in range(data.shape[1]-1):
                self.condition[state][i] = {0:self.smooth,1:self.smooth}
        print("begin to comput conditional probability")
        for i in range(data.shape[0]):
            y = data[i][-1]#获得第i个用例的分类
            for j in range(data.shape[1]-1):
                self.condition[y][j][data[i][j]] = self.condition[y][j][data[i][j]]+1
        print("begin to normalize")
        #normalize
        for y in self.condition:
            for x in self.condition[y]:
                total = freq[y]+self.smooth*2
                #       获得y类的所有对象     对y类x属性的每个值进行平滑，每个属性2个属性值
                for a in self.condition[y][x]:
                    #print(self.condition[y][x][a],total)
                    #print(y,x,a)
                    self.condition[y][x][a] = log2(self.condition[y][x][a])-log2(total)
                
    def predictor(self,data):
        Y = data[:,-1]
        res = []
        for i in range(Y.shape[0]):
            result = None
            max = -float("inf")
            for y in self.prior:
                prob = self.prior[y]
                for x in self.condition[y]:
                    #print(y,x,data[i,x])    
                    prob += self.condition[y][x][data[i,x]]
                (max, result) = (prob, y) if prob>max else (max, result)
                #print("\n"+str(i))
                #print("class:"+str(y)+" probability"+str(prob),end="")
            assert result!=None
            res.append(result)
        return res
                    
        
