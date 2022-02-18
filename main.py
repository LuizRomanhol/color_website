from flask import Flask, render_template, request, redirect, url_for
from flask import session
import numpy as np
import cv2
import random as rnd
import src.color_calibration as color
import flask

app = Flask(__name__)
app.config['SECRET_KEY'] = 'password'

@app.route('/', methods=['GET', 'POST'])
def index():
	
    session["refpath"] = ""
    session["imgpaths"] = []
    session["points"] = ""
    session["results"] = []
    session["hidden"] = "hidden"
    
    def save_img(img):
        filepath = "static/tmp/"+str(rnd.randint(0,1000000))+".jpeg"
        cv2.imwrite(filepath, img)
        return filepath
        
    if request.method == "POST":

        reffile =  request.files['ref']
        imgfiles =  flask.request.files.getlist("imgs")
        ref = cv2.imdecode(np.fromfile(reffile, np.uint8), cv2.IMREAD_COLOR)
        imgs = []

        for imgfile in imgfiles:
	        img = cv2.imdecode(np.fromfile(imgfile, np.uint8), cv2.IMREAD_COLOR)
        	session["imgpaths"].append(save_img(img))
        	imgs.append(img)
        	
        session["refpath"] = save_img(ref)
        coords = color.get_coords(session["refpath"], template_path = "color_website/static/images/template.jpg",template_coords = [(17,87),(42,87),(67,87),(91,87),(116,87),(140,87)])
        session["points"] = save_img(color.draw_circles(ref.copy(), coords))
        imgs = color.calibrate(ref.copy(),imgs, coords)
        
        for img in imgs:
        	session["results"].append(save_img(img))

        session["hidden"] = "show"
               
    return render_template('index.html', points = session["points"][13:], results = session["results"], hidden = session["hidden"])
    

#if __name__ == '__main__':
#    app.run(host='0.0.0.0', port=8875)


