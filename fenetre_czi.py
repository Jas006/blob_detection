from fenetre_accueil import *
import webbrowser
nombre_de_canaux=alpha.canal
nombre_de_zstack=alpha.entry10


def callback(url):
    webbrowser.open_new(url)
from tkinter import filedialog

class fenetre_analyse(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.liste=[]
        self.liste2=[]
        self.liste_canal=[]
        self.liste3 = []
        self.liste4 = []
        self.liste_z = []
        self.label_check=tk.Label(text="Séléctionner les canaux à analyser").pack()
        for i in range(nombre_de_canaux):
            self.liste.append(tk.IntVar())
        for i in range(nombre_de_canaux):
            self.liste2.append(tk.Checkbutton(text=f"Canal{i + 1}", variable=self.liste[i]))
            self.liste2[i].pack()
        self.label_check_z = tk.Label(text="Séléctionner les z_stack à analyser").pack()
        for i in range(nombre_de_zstack):
            self.liste3.append(tk.IntVar())
        for i in range(nombre_de_zstack):
            self.liste4.append(tk.Checkbutton(text=f"z_stack{i + 1}", variable=self.liste3[i]))
            self.liste4[i].pack()

        self.label_filtre_gauss=tk.Label(text=f"Séléctionner le sigma du filtre gaussien (3)")
        self.label_filtre_gauss.pack()
        self.label_filtre_gauss.bind("<Button-1>", lambda e: callback("https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm"))
        self.sigma_gauss=tk.Entry()
        self.sigma_gauss.pack()
        self.label_wth = tk.Label(text=f"Séléctionner la taille du filtre white top hat(30)")
        self.label_wth.pack()
        self.label_wth.bind("<Button-1>",
                                     lambda e: callback("https://en.wikipedia.org/wiki/Top-hat_transform"))
        self.wth = tk.Entry()
        self.wth.pack()
        self.label_seuil=tk.Label(text="Choisir le percentile minimum des points à conserver (99.2)")
        self.label_seuil.pack()
        self.seuil=tk.Entry()
        self.seuil.pack()
        self.label_blob=tk.Label(text="Séléctiionner les paramètres du blob detection:")
        self.label_blob.pack()
        self.label_blob.bind("<Button-1>",lambda e: callback("https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log"))
        self.label_max=tk.Label(text="Entrer la valeur du max sigma (30)")
        self.label_max.pack()
        self.max=tk.Entry()
        self.max.pack()
        self.label_num = tk.Label(text="Entrer la valeur du num sigma (10)")
        self.label_num.pack()
        self.num = tk.Entry()
        self.num.pack()
        self.thresh = tk.Entry()
        self.label_thresh = tk.Label(text="Entrer la valeur du threshold (0.1)")
        self.label_thresh.pack()
        self.thresh.pack()
        self.label_distance=tk.Label(text="Sélectionner la distance maximale entre les points semblables (3)")
        self.label_distance.pack()
        self.distance=tk.Entry()
        self.distance.pack()
        self.bouton0=tk.Button(text='parcourir',command=self.save)
        self.label_bouton_0=tk.Label(text="Séléctionner un dossier d'enregistrement des analyses")
        self.label_bouton_0.pack()
        self.bouton0.pack()
        self.bouton1=tk.Button(text="Continuer analyse",command=self.continuer)
        self.bouton1.pack()

    def save(self):
        self.filename = filedialog.askdirectory()
        return self.filename
    def canal_oui(self,i):
        self.liste_canal.append(i)
    def continuer(self):
        for i in range(nombre_de_canaux):
            if self.liste[i].get()==1:
                self.liste_canal.append(i)
        for i in range(nombre_de_zstack):
            if self.liste3[i].get()==1:
                self.liste_z.append(i)
        self.sigma_gauss=int(self.sigma_gauss.get())
        self.wth = int(self.wth.get())
        self.seuil = float(self.seuil.get())
        self.max = int(self.max.get())
        self.num = int(self.num.get())
        self.thresh = float(self.thresh.get())
        self.distance=int(self.distance.get())
        return self.sigma_gauss,self.wth,self.liste_canal,self.seuil,self.max,self.num,self.thresh,self.liste_canal,self.liste_z

beta=fenetre_analyse()
beta.title("Fenêtre czi")
