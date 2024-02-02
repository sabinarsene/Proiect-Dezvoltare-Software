import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score

def evaluare_model(y, predictie, clase):
    c = confusion_matrix(y, predictie)
    acuratete = np.diag(c)*100/np.sum(c,axis=1)
    acuratete_medie = np.mean(acuratete)
    acuratete_globala = sum(np.diag(c))*100/len(y)
    kappa = cohen_kappa_score(y, predictie)
    t_conf = pd.DataFrame(c, clase, clase)
    t_conf["Acuratete"] = acuratete
    t_acuratete = pd.Series([acuratete_globala, acuratete_medie, kappa], ["Acuratete globala", "Acuratete medie", "Index Cohen-Kappa"],
                            name="Acuratete")
    return t_acuratete, t_conf