from fenetre_accueil import *
from blob_detection import *


alpha.mainloop()
typ=alpha.type_file
if typ=='czi':
    from fenetre_czi import *
    beta.mainloop()
    complet_czi(path=alpha.path, channel=beta.liste_canal, sigma_filtre_gaussien=int(beta.sigma_gauss),
                taille_tophat=int(beta.wth), seuil_percentile=float(beta.seuil), max_sigma=int(beta.max), num_sigma=int(beta.num),
                threshold=float(beta.thresh), Path_enregistrement=beta.filename, d_semblables=int(beta.distance))




