import tkinter as tk
from tkinter import filedialog

class fenetre_accueil(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.entry1 = tk.Entry(self)
        self.button0=tk.Button(text="parcourir",command=self.browse)
        self.label1 = tk.Label(text="Entrez votre chemin d'accès:")
        self.button1 = tk.Button(self, text="Continuer", command=self.on_button)
        self.label3=tk.Label(text='Votre image comporte-t-elle des ROIs?')
        self.check_o1=tk.Checkbutton(text='Oui',command=self.ROI_oui)
        self.check_n1=tk.Checkbutton(text='Non',command=self.ROI_non)
        self.label4=tk.Label(text='Combien de canaux comporte votre image?')
        self.entry2=tk.Entry(self)
        self.label5=tk.Label(text='Votre image est-elle un z stack?')
        self.check_o2 = tk.Checkbutton(text='Oui',command=self.z_oui)
        self.check_n2 = tk.Checkbutton(text='Non',command=self.z_non)
        self.label1.pack()
        self.button0.pack()
        self.label3.pack()
        self.check_o1.pack()
        self.check_n1.pack()
        self.label4.pack()
        self.entry2.pack()
        self.label5.pack()
        self.check_o2.pack()
        self.check_n2.pack()
        self.button1.pack()
        self.type_file=None
        self.Roi='non'
        self.z_stack='non'

    def browse(self):
        self.f = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        return self.f
    def ROI_oui(self):
        self.Roi='oui'
    def ROI_non(self):
        self.Roi='non'
    def z_oui(self):
        self.z_stack='oui'
    def z_non(self):
        self.z_stack='non'
    def on_button(self):
        self.path = self.f
        self.canal=self.entry2.get()
        self.canal=int(self.canal)
        self.type_file = self.path[-3] + self.path[-2] + self.path[-1]
        self.destroy()
        return self.type_file,self.canal,self.path

alpha = fenetre_accueil()
alpha.title('fenetre d\'accueil')
