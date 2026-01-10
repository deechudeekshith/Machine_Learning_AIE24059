#q1:
def pair_count(values,target=10):
    total_pairs=0
    for i in range(len(values)):
        for j in range(i+1,len(values)):
            if values[i]+values[j]==target:
                total_pairs +=1
    return total_pairs
nums=[2,7,4,1,3,6]
print("Q1:")
print("pairs count:",pair_count(nums))
#q2:
def range(values):
    if len(values)<3:
        return "range determination not possible"
    highest =max(values)
    lowest =min(values)
    return highest -lowest
nums=[5,3,8,1,0,4]
print("Q2:")
print("range:",range(nums))

#q3:
import numpy as np

def matrix_power(mat,power):
    mat=np.array(mat)
    return np.linalg.matrix_power(mat,power)
matrix=[
    [1,2],[3,4]
    ]
print("Q3:")
print("Matrix power:")
print(matrix_power(matrix,2))

#q4:
def highest_freq(text_val):
    char_count={}
    top_char=None
    top_count=0

    for ch in text_val:
        if ch.isalpha():
            char_count[ch]=char_count.get(ch,0)+1
            if char_count[ch]>top_count:
                top_count=char_count[ch]
                top_char=ch
    return top_char,top_count
text_data="hippopotamus"
char,freq=highest_freq(text_data)
print("Q4")
print(f"most frequence char:{char},count:{freq}")
#q5
import numpy as np
rng = np.random.default_rng(seed = 42)
d = rng.integers(1,10,size = 25)

def mean_median(d):
    mean = np.mean(d)
    median = np.median(d)
    max_count = 0
    mode = None
    for num in d:
        if list(d).count(num)>max_count:
            max_count = list(d).count(num)
            mode = num
    return mean,median,mode
mean,median,mode = mean_median(d)
print("Q5:")
print(d)
print(f"mean: {mean}, median:{median}, mode : {mode}")

