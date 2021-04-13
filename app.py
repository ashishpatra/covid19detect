from flask import Flask, render_template, request
import pickle
app = Flask(__name__)

# open a file, where you ant to store the data
file = open('model.pkl', 'rb')

lrm = pickle.load(file)
file.close()

@app.route('/', methods=["GET","POST"])
def home():
    if request.method == 'POST':
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        runnyNose = int(myDict['runnyNose'])
        diffBreath = int(myDict['diffBreath'])
        
        # Code for Inference
        inputFeatures = [fever,pain,age,runnyNose,diffBreath]
        infProb = lrm.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template('view.html', inf=round(infProb*100))
    return render_template('index.html')
    # return 'Hello, World!' + str(infProb)
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == "__main__":
    app.run(debug=True)