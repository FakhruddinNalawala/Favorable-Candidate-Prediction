import csv
import re

predicted=open("predicted_winners.csv","r")
winners=open("Winners.csv","r")

accuracy=0
total=28

for predict in predicted:
    # total+=1
    # predict.strip()
    const_predict=predict.split(",")[0].lower()
    cand_predict=predict.split(",")[1].lower()
    for win in winners:
        # win.strip()
        const_win=win.split(",")[0].lower()
        cand_win=win.split(",")[1].lower()
        print("predicted: ",cand_predict," winner: ",cand_win," const pred  is ",const_predict," const win is ",const_win)
        if(const_predict==const_win and ((cand_win in cand_predict) or ((cand_predict in cand_win)))):
            accuracy+=1
            break
        else:
            break

print("accuracy is ",accuracy/total)         
        
