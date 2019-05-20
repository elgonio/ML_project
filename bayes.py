import csv

numWins = 0
numLoses = 0
characterWin = [0] * 113
characterLoss = [0] * 113
characterWinNonChanged = [0] * 113
characterLossNonChanged = [0] * 113
characterArray = []
win_loss_stat = []
probabWin1 = 1
probabLoss1 = 1
probabWin2 = 1
probabLoss2 = 1
accurate = 0
accurate2 = 0
totalInstances = 0
testClean = []
with open('dota2Train.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    myList = list(reader)

with open('dota2Test.csv', 'r') as csvFile2:
    reader2 = csv.reader(csvFile2)
    test2 = list(reader2)
# so i need to get the characters into winners and losers
# so i need a count of how many wins vs loses
# so each hero needs to have how many wins it gets next to it
# there should be one main array
for row in myList:
    characterArray.append(row[4:])
    win_loss_stat.append(row[0])
#print(test2)
for row in test2:
    testClean.append(row[4:])
    #print(testClean)
# print(len(characterArray))
for row in range(len(characterArray)):
    for point in range(len(characterArray[0])):
        if int(characterArray[row][point]) == -1 and int(myList[row][0]) == -1:  # nega win
            characterWin[int(point)] = int(characterWin[int(point)]) + 1

        elif int(characterArray[row][point]) == 1 and int(myList[row][0]) == 1:  # posi win
            characterWin[int(point)] = int(characterWin[int(point)]) + 1

        elif int(characterArray[row][point]) == -1 and int(myList[row][0]) == 1:  # nega loss
            characterLoss[int(point)] = int(characterLoss[int(point)]) + 1

        elif int(characterArray[row][point]) == 1 and int(myList[row][0]) == -1:  # posi loss
            characterLoss[int(point)] = int(characterLoss[int(point)]) + 1
characterWinNonChanged = list(characterWin)
characterLossNonChanged = list(characterLoss)
'''for row in range(len(characterWin)):
    print(row)
    print("Win: " + str(characterWin[row]) + " Loss: " + str(characterLoss[row]))'''
# now the probablity equation
# actually need a list of all the wins relative to losses but for every game theres a loser and a winner
for row in range(len(testClean)):
    for point in range(len(testClean[row])):
        if int(testClean[row][point]) == 1:
            if int(characterWin[point]) == 0 or int(characterLoss[point]) == 0:
                var = [item + 1 for item in characterWin]
                characterWin = list(var)
                var1 = [item + 1 for item in characterLoss]
                characterLoss = list(var1)
                probabWin1 = (characterWin[point] / len(myList)) * probabWin1
                probabLoss1 = (1 - (characterLoss[point] / len(myList)) * probabLoss1)
            else:
                probabWin1 = (characterWin[point]/len(myList)) * probabWin1
                probabLoss1 = (1-(characterLoss[point]/len(myList)) * probabLoss1)
        elif int(testClean[row][point]) == -1:
            if int(characterWin[point]) == 0 or int(characterLoss[point]) == 0:
                var = [item + 1 for item in characterWin]
                characterWin = list(var)
                var1 = [item +1 for item in characterLoss]
                characterLoss = list(var1)
                probabWin1 = (1 - (characterWin[point] / len(myList)) * probabWin1)
                probabLoss1 = (characterLoss[point] / len(myList)) * probabLoss1
            else:
                probabWin1 = (1 - (characterWin[point] / len(myList)) * probabWin1)
                probabLoss1 = (characterLoss[point] / len(myList)) * probabLoss1

    probabW = probabWin1 / (probabWin1 + probabLoss1)
    probabL = probabLoss1 / (probabWin1 + probabLoss1)
    print("Prob win: "+str(probabW))  # probab Radiant Win
    print("Prob loss: "+str(probabL))  # probab Radaint Loss
    if probabW >= 0.5 and int(test2[row][0])==1:
        accurate = accurate + 1
    if probabW < 0.5 and int(test2[row][0])==-1:
        accurate = accurate + 1
    probabWin1 = 1
    probabLoss1 = 1
'''probabW = probabWin/(probabWin+probabLoss)
probabL = probabLoss/(probabWin+probabLoss)
print(probabW)#probab Radiant Win
print(probabL)#probab Radaint Loss'''
totalInstances = len(testClean)
'''print(characterWinNonChanged)
print(len(characterWinNonChanged))
print(characterLossNonChanged)
print(len(characterLossNonChanged))'''
print(totalInstances)
print(accurate)
print(accurate/totalInstances)
csvFile.close()
csvFile2.close()
