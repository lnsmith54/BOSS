import sys
import os
import numpy as np
import glob
#import math


'''
Reads all the files in the directory and extracts a summary of the results. 
Expects 4 runs of each scenario
Writes the summary to a file Summary.
The highlights are the Class accuracies for Test and best_test

Typical input at the end of the results files looks like: 

Class accuracies for Test:  {'test': [68.0, 98.6, 19.3, 0.0, 92.7, 96.8, 94.39999999999999, 74.2, 95.5, 92.9]}
Class accuracies for the top 5 and top 20 of Test:  {'test': [[80.0, 100.0, 20.0, 0.0, 100.0, 100.0, 100.0, 60.0, 100.0, 100.0], [95.0, 100.0, 25.0, 0.0, 100.0, 100.0, 10\
0.0, 65.0, 100.0, 100.0]]}
kimg 32448  accuracy train/valid/test/best_test  100.00  75.58  74.47  76.42
kimg 32512  accuracy train/valid/test/best_test  100.00  75.42  74.52  76.42
kimg 32576  accuracy train/valid/test/best_test  100.00  75.42  74.58  76.42
kimg 32640  accuracy train/valid/test/best_test  100.00  75.54  74.67  76.42
Class accuracies for Test:  {'test': [66.4, 97.6, 19.7, 0.0, 96.1, 97.5, 97.1, 76.7, 98.3, 96.89999999999999]}
Class accuracies for the top 5 and top 20 of Test:  {'test': [[80.0, 100.0, 40.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], [80.0, 100.0, 30.0, 0.0, 100.0, 100.0, 1\
00.0, 75.0, 100.0, 100.0]]}
Number of training pseudo-labels in each class:  [  377 49608     9     5]  for classes:  [0 1 8 9]
kimg 32704  accuracy train/valid/test/best_test  100.00  75.56  74.63  76.42


'''

#numFiles = int(sys.argv[2])
numFiles = 4

bestAcc = [0]*numFiles

print("=> Writing out files ....")
filename = 'ResultsSummary'
print(filename)
fileOut = open(filename,'w')

files = os.listdir('.')
listing = glob.glob('./*0')

listing.sort()
for j in range(len(listing)):
    classAcc = ['', '', '', '']
    bestAcc  = ['', '', '', '']
    numTrainPerClass  = ['', '', '', '']
#    print(listing[j])
    for i in range(0,numFiles):
        name = listing[j]
        name = name[:-1] + str(i)
#       print(name," exits ",os.path.isfile(name))
        if os.path.isfile(name):
            with open(name,"r") as f:
                for line in f:
                    if (line.find("lass accuracies for Test") > 0):
                        pref, post = line.split("[")
                        classAcc[i] = post[:-3]
#Number of training pseudo-labels in each class:  [ 4488  6828  1324  4870 13553  5727  6403  3551  1552  1703]  for classes:  [0 1 2 3 4 5 6 7 8 9]
#                    if (line.find("pseudo-labels") > 0):
#                        pref, mid,  post = line.split("[")
#                        numbers, post = mid.split("]")
#                        numTrainPerClass[i] = numbers

                    if (line.find("accuracy train/valid/test/best_test") > 0):
                        pref, post = line.split("best_test")
                        accs = post.split("  ")
                        bestAcc[i] = accs[4][:-1]

    print(name,"   ",bestAcc)
    for i in range(0,numFiles):
        print(classAcc[i])
#        print(numTrainPerClass[i])

#    fileOut.write('{:f}'.format(bestAcc)
    fileOut.write(name+"   ")
    for i in range(0,numFiles):
        fileOut.write(bestAcc[i]+"   ")
    fileOut.write("\n")
    for i in range(0,numFiles):
#        fileOut.write(classAcc[i])
        accs = classAcc[i].split(",")
        try:
            for x in accs:
                fileOut.write('{0:0.2f},  '.format(float(x)))
        except:
            pass
        fileOut.write("\n")
#    for i in range(0,numFiles):
#        fileOut.write(numTrainPerClass[i])
#        fileOut.write("\n")
fileOut.close()
exit(1)
