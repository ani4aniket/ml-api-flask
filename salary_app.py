import tkinter as tk

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 400, height = 300,  relief = 'raised')
canvas1.pack()

label1 = tk.Label(root, text='Salary Predictor')
label1.config()
canvas1.create_window(200, 25, window=label1)

label2 = tk.Label(root, text='Enter year of experience:')
label2.config()
canvas1.create_window(200, 100, window=label2)

entry1 = tk.Entry (root) 
canvas1.create_window(200, 140, window=entry1)


import pickle
import numpy as np
# load the model from disk
regressor = pickle.load(open("regressor.model", 'rb'))

sc_X = pickle.load(open("scaler_x.model", 'rb'))

sc_y = pickle.load(open("scaler_y.model", 'rb'))

def getSalary ():
    
    x1 = entry1.get()


    sample_input = np.array([x1]).reshape(-1, 1)
    sample_output = regressor.predict(sc_X.transform(sample_input))

    label3 = tk.Label(root, text= 'Expected Salary:')
    canvas1.create_window(200, 210, window=label3)
    
    label4 = tk.Label(root, text= round(float((sc_y.inverse_transform(sample_output))[0][0]), 2))
    canvas1.create_window(200, 230, window=label4)

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def restore_from_pickle():
    root.wm_title("Embedding in Tk")
    fig = Figure(figsize=(5, 4), dpi=100)
    with open('/Users/aniketkumar/Documents/mlwork/trainGraph.pickle', 'rb') as fid:
        fig = pickle.load(fid)
    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    return
 
def getGraph ():
    root.wm_title("Embedding in Tk")

    

    fig = Figure(figsize=(5, 4), dpi=100)
    t = np.arange(0, 3, .01)
    fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


def _quit():
    root.quit()     # stops mainloop
    root.destroy() 
   
button1 = tk.Button(text='Predict', command=getSalary)
canvas1.create_window(200, 180, window=button1)


# button = tk.Button(master=root, text="Show Graph", command=getGraph)
# button.pack(side=tk.LEFT)



button = tk.Button(master=root, text="Quit", command=_quit)
button.pack(side=tk.RIGHT)

button = tk.Button(master=root, text='Show Graph', command=restore_from_pickle)
button.pack(side=tk.LEFT)

root.mainloop()