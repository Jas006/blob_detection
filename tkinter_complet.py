from fenetre_accueil import *
from blob_detection_czi import *


alpha.mainloop()
typ=alpha.type_file
if typ=='czi':
    from fenetre_czi import *
    beta.mainloop()
    complet_czi(path=alpha.path,liste_z=beta.liste_z, channel=beta.liste_canal, sigma_filtre_gaussien=int(beta.sigma_gauss),
                taille_tophat=int(beta.wth), seuil_percentile=float(beta.seuil), max_sigma=int(beta.max), num_sigma=int(beta.num),
                threshold=float(beta.thresh), Path_enregistrement=alpha.dirname, d_semblables=int(beta.distance),enregistrementblob=alpha.dirname_blob_detection,enregistrement_blur=alpha.dirname_gauss_blur,enregistrement_hdome=alpha.dirname_h_dome,enregistrement_th=alpha.dirname_top_hat,enregistrement_seuil=alpha.dirname_seuil)




