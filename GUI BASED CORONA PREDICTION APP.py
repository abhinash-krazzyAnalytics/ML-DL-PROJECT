# -*- coding: utf-8 -*-
"""
Created on Sat May 16 16:46:43 2020

@author: Abhinash
"""

#modelling deployment on gui
#step 1 library for ml and gui
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from tkinter import*
from tkinter import messagebox
import pyttsx3

data=pd.DataFrame({"bodytemp":[98,102,101,98,102,100,100,103,101,
                              100,103,104,104,101,103,101,100,100,
                              102,98,98,102,99,98,100,103,102,98,104,103],
                  "age":[36,88,71,80,6,88,56,47,61,89,49,71,
                         10,70,30,49,94,24,51,49,45,92,76,37,
                         38,43,49,22,21,38],
                "breath_problem":[0,1,1,0,0,0,0,1,1,0,1,
                                  1,1,0,1,0,0,1,1,0,0,1,
                                  0,0,0,1,1,0,1,0],
                "running_nose":[0,1,1,1,1,0,0,1,1,0,1,1,
                                1,1,1,0,0,0,1,0,0,1,1,0,
                                1,0,1,0,1,1],
                  "body_pain":[0,1,1,1,1,0,0,1,0,1,1,1,1,
                               0,1,1,1,1,1,0,0,1,0,0,1,0,
                               1,0,1,1],
                   "Corona_infection":["no","yes","yes","yes",'yes','no','no',
                                        'yes','yes','no','yes','yes','yes','no',
                                        'yes','no','no','no','yes','no',
                                       'no','yes','no','no','no','no','yes',
                                       'no','yes','yes']})
#GUI
win=Tk()
win.geometry("400x600")
#frame title
win.title("Corona Virus Prediction")
win.configure(background="light green")
Label(win,text="CORONA VIRUS PREDICTION APP",
          font=('arial',15,'bold'),bg='light green',
          fg='black',relief="solid").pack()

Label(win,text="BodyTemp 'F",
      fg='black',relief='solid',width=18).place(x=50,y=80)
Label(win,text="Person Age",
      fg='black',relief='solid',width=18).place(x=50,y=120)
Label(win,text="Breath problem",
      fg='black',relief='solid',width=18).place(x=50,y=160)
Label(win,text="Running nose",
      fg='black',relief='solid',width=18).place(x=50,y=200)
Label(win,text=" Body pain",
      fg='black',relief='solid',width=18).place(x=50,y=240)
Label(win,text="RESULT ANALYSIS",
      fg='black',relief='solid',width=18).place(x=50,y=280)

Label(win,text="Variable Entry Info",
      fg='black',relief='solid',width=18).place(x=50,y=330)
#########################################################
#entry window for all info
bt=StringVar()
ag=StringVar()
bp=StringVar()
rn=StringVar()
bdypn=StringVar()
Entry(win,textvariable=bt).place(x=230,y=80)
Entry(win,textvariable=ag).place(x=230,y=120)
Entry(win,textvariable=bp).place(x=230,y=160)
Entry(win,textvariable=rn).place(x=230,y=200)
Entry(win,textvariable=bdypn).place(x=230,y=240)

def info():
    infowin=Tk()
    infowin.geometry("300x300")
    Label(infowin,text="Body pain in Degree Fer").pack()
    Label(infowin,text="Person age ").pack()
    Label(infowin,text="Body pain yes 1 no 0").pack()
    Label(infowin,text="Running Nose yes 1 no 0").pack()
    Label(infowin,text="body pain yes 1 no 0").pack()
    infowin.mainloop()
        
Button(win,text="INFO",width=15,command=info).place(x=230,y=330)

def ml_model():
     #data dep:Corona_infection
     #dataindp--all except corona
     le=LabelEncoder()
     data["Corona_infection"]=le.fit_transform(data["Corona_infection"])
     y=data["Corona_infection"]
     x=data.drop(["Corona_infection"],axis=1)
     #form model
     model=LogisticRegression()
     model.fit(x,y)
     #data grab from entry window
     temp=float(bt.get())
     Age=float(ag.get())
     breath_p=float(bp.get())
     run_n=float(rn.get())
     body_pn=float(bdypn.get())
     x_test=[temp,Age,breath_p,run_n,body_pn]
     y_pred=model.predict([x_test,])
     y_ref={0:"Not Infected",1:"Yes Infected"}
     ref=y_pred[0]
     if(ref==0):
         Label(win,text=str(y_ref[ref]),fg="green").place(x=230,y=280)
         engine = pyttsx3.init()
         voices = engine.getProperty('voices')
         rate = engine.getProperty('rate')
         engine.setProperty('rate', rate-100)
         engine.say('its a happy news you are not effected by corona')
         engine.runAndWait()
     else:
         Label(win,text=str(y_ref[ref]),fg="red").place(x=230,y=280)
         engine =pyttsx3.init()
         voices = engine.getProperty('voices')
         rate = engine.getProperty('rate')
         engine.setProperty('rate', rate-100)
         engine.say('Bad news you are infected by corona contact Emergency number 111011 helpline')
         engine.runAndWait()
def terminate():
    win.destroy()
Button(win,text="Prediction",width=15,bg="yellow",command=ml_model).place(x=70,y=400)
Button(win,text="Termination",width=15,bg="yellow",command=win.destroy).place(x=200,y=400)
win.resizable(0,0)
win.mainloop() 