from flask import Flask, render_template, request
from pip import main
import pickle

# from requests import post, request

app = Flask(__name__)

file = open('model.pkl','rb')
clf=pickle.load(file)
file.close()

@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        Fever = int(myDict['Fever'])
        Age = int(myDict['Age'])
        Pain = int(myDict['Pain'])
        RunnyNose = int(myDict['RunnyNose'])
        DiffBreath = int(myDict['DiffBreath'])
        
        # Code for inference
        inputFeatures=[Fever, Pain, Age, RunnyNose, DiffBreath]
        InfProb=clf.predict_proba([inputFeatures])[0][1]
        print(InfProb)
        return render_template('show.html', inf=round(InfProb*100))
    return render_template('index.html')
    #return "<p>Hello, World!</p>" +  str(infProb)


if __name__ =="__main__":
    app.run(debug=True)