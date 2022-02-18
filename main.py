from flask import Flask, render_template, request, redirect, url_for
from flask import session
import numpy as np
import cv2
import random as rnd
import src.color_calibration as color

app = Flask(__name__)
app.config['SECRET_KEY'] = 'password'

@app.route('/', methods=['GET', 'POST'])
def index():
	
    session["imgpath"] = ""	
    session["points"] = ""
    session["result"] = ""
    session["hidden"] = "hidden"
	
    def save_img(img):
        filepath = "static/tmp/"+str(rnd.randint(0,1000000))+".jpeg"
        cv2.imwrite(filepath, img)
        return filepath
        
    if request.method == "POST":
        imagefile =  request.files['img']
        npimg = np.fromfile(imagefile, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        print(img.shape)
        session["imgpath"] = save_img(img)
        coords = color.get_coords(session["imgpath"], template_path = "static/images/template.jpg",template_coords = [(17,87),(42,87),(67,87),(91,87),(116,87),(140,87)])
        session["points"] = save_img(color.draw_circles(img.copy(), coords))
        session["result"] = save_img(color.calibrate(img.copy(), coords))
        session["hidden"] = "show"
        #print(session["imgpath"])
               
    return render_template('index.html', imgpath = session["imgpath"], points = session["points"], result = session["result"], hidden = session["hidden"])
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8875)


