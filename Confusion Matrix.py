

class PredictionMatrix:
    def precession(TP,FP):
        precession = TP/(TP+FP)
        print("The Precession is",precession)
        return
    def recall(TP,FN):
        recall = TP/(TP+FN)
        print("The recall is ",recall)
        return
print("Enter TP value")
TP = int(input())
print("Enter FN value")
FN = int(input())
print("Enter FP value")
FP = int(input())
print("Enter TN value")
TN = int(input())
PredictionMatrix.precession(TP,FP)
PredictionMatrix.recall(TP,FN)