# import requirements needed
from flask import Flask, render_template, request, url_for, send_file

from utils import get_base_url
from train import our_model

# global variables
prediction = None
model = None



# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 15000
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in deveflopment, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

@app.route(f'{base_url}')
def home():
    return render_template('index.html')

@app.route(f'{base_url}data_analysis')
def dataanalysis():
    return render_template('data_analysis.html')

@app.route(f'{base_url}models')
def models_():
    return render_template('models.html')

@app.route(f'{base_url}model_analysis')
def modelanalysis():
    return render_template('model_analysis.html')

@app.route(f'{base_url}team')
def team_():
    return render_template('team.html')

# set up the routes and logic for the webserver
@app.route(f'{base_url}interactive', methods = ["GET", "POST"])
def interactive_():
    global model
    global prediction

    # get input columns from html page
    if request.method == 'POST':
        phones = request.form.get("phones")
        infant_mortality = request.form.get("infant_mortality")
        birthrate = request.form.get("birthrate")
        deathrate = request.form.get("deathrate")
        net_migration = request.form.get("net_migration")
        coastline = request.form.get("coastline")
        agriculture = request.form.get("agriculture")
        industry = request.form.get("industry")
        service = request.form.get("service")
        arable = request.form.get("arable")
        crops = request.form.get("crops")
    # get prediction using using these features
        prediction = int(model.predict(phones, infant_mortality, birthrate, deathrate, net_migration, coastline, agriculture, industry, service, arable, crops))

    # return prediction

    return render_template('interactive.html', prediction = prediction)

    # out html page will ask users to input these features, then it will display the prediction 
    # {prediction}

# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    model = our_model()
    app.run(host = '0.0.0.0', port=port, debug=True)
