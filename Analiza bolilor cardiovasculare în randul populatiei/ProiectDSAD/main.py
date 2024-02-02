from functii import *
from grafice import *
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import f
from sklearn.naive_bayes import GaussianNB

set_antrenare_testare = pd.read_csv("Cardiovascular_Disease.csv", index_col=0)
variabile = list(set_antrenare_testare)
predictori = variabile[:-1]
tinta = variabile[-1]

# Splitare/impartire set antrenare-testare
x_train, x_test, y_train, y_test = (train_test_split(set_antrenare_testare[predictori],set_antrenare_testare[tinta],test_size=0.4))
#print(x_train,y_train)

# Construire model liniar
model_lda = LinearDiscriminantAnalysis()
model_lda.fit(x_train,y_train)

# Calculul puterii de discriminare a predictorilor
G = model_lda.means_
clase = model_lda.classes_

n_train = len(x_train)
q = len(clase)
m = len(predictori)
T = np.cov(x_train.values, rowvar=False) # T matricea de covarianta totala
x_train_ = np.mean(x_train.values, axis=0)
DG = np.diag(model_lda.priors_)
B = (G - x_train_).T @ DG @ (G - x_train_) # matricea de covarianta inter-clase
W = T - B # matricea de covarianta intra-clase

# Putere discriminare predictori
f_predict = (np.diag(B) / (q - 1)) / (np.diag(W) / (n_train - q))
#print(f_predict)

# Test Fisher
p_values = 1 - f.cdf(f_predict, q - 1, n_train - q)
#print(p_values)

t_predictori = pd.DataFrame(
    {
        "Predictori": predictori,
        "Putere discriminare": f_predict,
        "P_Values": p_values
    })
t_predictori.to_csv("Predictori.csv", index=False)

# Analiza discriminatori pe setul de testare
nr_disc = min(q - 1, m)

# Preluare scoruri discriminante
z = model_lda.transform(x_train)
#print(nr_disc)
etichete_z = ["z" + str(i + 1) for i in range(nr_disc)]

# Calcul putere discriminare discriminatori pe setul de antrenare
T_z = np.cov(z,rowvar=False)
Z_ = np.mean(z,axis=0) # T matricea de covarianta totala
G_z = model_lda.transform(G)
DG = np.diag(model_lda.priors_)
B_z = (G_z - Z_).T @ DG @ (G_z - Z_) # matricea de covarianta inter-clase
W_z = T_z - B_z # matricea de covarianta intra-clase
f_discriminatori = (np.diag(B_z) / (q - 1)) / (np.diag(W_z) / (n_train - q))

# Test Fisher
p_value_z = 1 - f.cdf(f_discriminatori, q - 1, n_train - q)
#print(p_value_z)

t_discrim = pd.DataFrame(
    {
        "Putere discriminare": f_discriminatori,
        "P_values": p_value_z
    }, etichete_z
)
t_discrim.to_csv("Discriminatori.csv",index_label="Discriminatori")

# Salvare scoruri discriminante
tz = pd.DataFrame(z, x_train.index, etichete_z)
tz.to_csv("z.csv")

# Vizualizare distributie discriminatori
for i in range(nr_disc):
    plot_distributie(z, y_train, i)

# Calcul centrii discriminatori
zg = tz.groupby(by=y_train.values).mean().values

# Vizualizare instante si centrii după axele discriminante
for i in range(nr_disc - 1):
    for j in range(i + 1, nr_disc):
        scatterplot(z, zg, y_train, clase, i, j)

# Testare model
#lda = Linear Discriminant Analysis
predictie_test_lda = model_lda.predict(x_test)
t_acuratete_lda, t_conf_lda = evaluare_model(y_test, predictie_test_lda, clase)
t_acuratete_lda.to_csv("Acuratete_lda.csv", index_label="Tip")
t_conf_lda.to_csv("MatriceaDeConfuzie_lda.csv", index_label="Target")

# Calcul Erori
t_err = pd.DataFrame(
    {
        tinta:y_test
    }, x_test.index)
t_err["Predictie LDA"] = predictie_test_lda
#print(t_err)

# Aplicare model
set_aplicare = pd.read_csv("Cardiovascular_Disease_no_target.csv", index_col=0)
predictie_lda = model_lda.predict(set_aplicare[predictori])
set_aplicare["Predictie LDA"] = predictie_lda

# Modelul bayesian
model_bayes = GaussianNB()
model_bayes.fit(x_train, y_train)

# Testare
predictie_test_bayes = model_bayes.predict(x_test)
t_acuratete_bayes, t_conf_bayes = evaluare_model(y_test, predictie_test_bayes, clase)
t_acuratete_bayes.to_csv("Acuratete_bayes.csv", index_label="Tip")
t_conf_bayes.to_csv("MatriceaDeConfuzie_bayes.csv", index_label="Target")

t_err["Predictie Bayes"] = predictie_test_bayes

# Aplicare Model
predictie_bayes = model_bayes.predict(set_aplicare[predictori])
set_aplicare["Predictie Bayes"] = predictie_bayes

set_aplicare.to_csv("Predictii.csv")

# Analiza erori
t_err_lda = t_err[predictie_test_lda != y_test]
t_err_bayes = t_err[predictie_test_bayes != y_test]
t_err_lda.to_csv("Err_lda.csv")
t_err_bayes.to_csv("Err_bayes.csv")

show()