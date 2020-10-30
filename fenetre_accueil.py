import tkinter as tk
from tkinter import filedialog
import ntpath
import os

text_accueil=open("text_file_accueil.txt","r")
path=text_accueil.readlines()[0]
text_accueil.close()


class fenetre_accueil(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.f=''
        self.entry1 = tk.Entry(self)
        self.button0=tk.Button(text="parcourir",command=self.browse)
        self.label1 = tk.Label(text="Entrer le chemin d'accès:")
        self.button1 = tk.Button(self, text="Continuer", command=self.on_button)
        self.label3=tk.Label(text='Afficher l\'image originale dans le fichier d\'analyse')
        self.check_o1=tk.Checkbutton(text='Oui',command=self.image1_oui)
        self.check_n1=tk.Checkbutton(text='Non',command=self.image1_non)
        self.label1.pack()
        self.button0.pack()
        self.label3.pack()
        self.check_o1.pack()
        self.check_n1.pack()
        self.button1.pack()
        self.type_file=None
        self.image1='non'

    def browse(self):
        self.f = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        return self.f
    def image1_oui(self):
        self.image1_on='oui'
    def image1_non(self):
        self.image1_on='non'
    def on_button(self):
        if self.f=='':
            self.path=path
        else:
            self.path=self.f
        self.type_file = self.path[-3] + self.path[-2] + self.path[-1]
        self.nom_dossier = ntpath.basename(self.path)
        self.nom_dossier = self.nom_dossier[:-4]
        self.dirname=r"analyse_" + self.nom_dossier
        if not os.path.exists(self.dirname):
            self.path_enregistrement=os.mkdir(self.dirname)
        else:
            self.path_enregistrement=self.dirname
        if self.image1_on=="oui":
            self.dirname_image1 = r"analyse_" + self.nom_dossier + r"\00_image_initiale"
            if not os.path.exists(self.dirname_image1):
                self.image1 = os.mkdir(self.dirname_image1)
            else:
                self.image1 = self.dirname_image1
        else:
            pass
        self.dirname_gauss_blur = r"analyse_" + self.nom_dossier+r"\01_gaussian_blur"
        if not os.path.exists(self.dirname_gauss_blur):
            self.path_enregistrement = os.mkdir(self.dirname_gauss_blur)
        else:
            self.path_enregistrement = self.dirname_gauss_blur
        self.dirname_h_dome = r"analyse_" + self.nom_dossier + r"\02_h_dome"
        if not os.path.exists(self.dirname_h_dome):
            self.path_enregistrement = os.mkdir(self.dirname_h_dome)
        else:
            self.path_enregistrement = self.dirname_h_dome
        self.dirname_top_hat = r"analyse_" + self.nom_dossier + r"\03_top_hat"
        if not os.path.exists(self.dirname_top_hat):
            self.path_enregistrement = os.mkdir(self.dirname_top_hat)
        else:
            self.path_enregistrement = self.dirname_top_hat
        self.dirname_seuil = r"analyse_" + self.nom_dossier + r"\04_seuil"
        if not os.path.exists(self.dirname_seuil):
            self.path_enregistrement = os.mkdir(self.dirname_seuil)
        else:
            self.path_enregistrement = self.dirname_seuil
        self.dirname_blob_detection = r"analyse_" + self.nom_dossier + r"\05_blob_detection"
        if not os.path.exists(self.dirname_blob_detection):
            self.path_enregistrement = os.mkdir(self.dirname_blob_detection)
        else:
            self.path_enregistrement = self.dirname_blob_detection
        self.dirname_image_detection = r"analyse_" + self.nom_dossier + r"\06_image_detection"
        if not os.path.exists(self.dirname_image_detection):
            self.path_detection = os.mkdir(self.dirname_image_detection)
        else:
            self.path_detection = self.dirname_image_detection
        self.destroy()
        text_file=open("text_file_accueil.txt","w")
        text_file.write(self.path)
        text_file.close()
        return self.type_file,self.path,self.image1_on,self.dirname_image1,self.dirname_image_detection

alpha = fenetre_accueil()
alpha.title('fenetre d\'accueil')

