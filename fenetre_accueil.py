import tkinter as tk
from tkinter import filedialog
import ntpath
import os

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
        self.label10 = tk.Label(text='Combien de z_stack comporte votre image?')
        self.entry10 = tk.Entry(self)
        self.label1.pack()
        self.button0.pack()
        self.label3.pack()
        self.check_o1.pack()
        self.check_n1.pack()
        self.label4.pack()
        self.entry2.pack()
        self.label10.pack()
        self.entry10.pack()
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
        self.entry10=int(self.entry10.get())
        self.type_file = self.path[-3] + self.path[-2] + self.path[-1]
        self.nom_dossier = ntpath.basename(self.path)
        self.nom_dossier = self.nom_dossier[:-4]
        self.dirname=r"analyse_" + self.nom_dossier
        if not os.path.exists(self.dirname):
            self.path_enregistrement=os.mkdir(self.dirname)
        else:
            self.path_enregistrement=self.dirname
        self.dirname_gauss_blur = r"analyse_" + self.nom_dossier+r"\gaussian_blur"
        if not os.path.exists(self.dirname_gauss_blur):
            self.path_enregistrement = os.mkdir(self.dirname_gauss_blur)
        else:
            self.path_enregistrement = self.dirname_gauss_blur
        self.dirname_h_dome = r"analyse_" + self.nom_dossier + r"\h_dome"
        if not os.path.exists(self.dirname_h_dome):
            self.path_enregistrement = os.mkdir(self.dirname_h_dome)
        else:
            self.path_enregistrement = self.dirname_h_dome
        self.dirname_top_hat = r"analyse_" + self.nom_dossier + r"\top_hat"
        if not os.path.exists(self.dirname_top_hat):
            self.path_enregistrement = os.mkdir(self.dirname_top_hat)
        else:
            self.path_enregistrement = self.dirname_top_hat
        self.dirname_seuil = r"analyse_" + self.nom_dossier + r"\seuil"
        if not os.path.exists(self.dirname_seuil):
            self.path_enregistrement = os.mkdir(self.dirname_seuil)
        else:
            self.path_enregistrement = self.dirname_seuil
        self.dirname_blob_detection = r"analyse_" + self.nom_dossier + r"\blob_detection"
        if not os.path.exists(self.dirname_blob_detection):
            self.path_enregistrement = os.mkdir(self.dirname_blob_detection)
        else:
            self.path_enregistrement = self.dirname_blob_detection
        self.destroy()
        return self.type_file,self.canal,self.path,self.entry10

alpha = fenetre_accueil()
alpha.title('fenetre d\'accueil')
