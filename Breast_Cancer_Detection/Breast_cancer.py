import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("cancer.csv")

data = data.rename(columns = {"diagnosis":"target"})

sns.countplot(data["target"])
print(data.target.value_counts())

data["target"] = [1 if i.strip() == "M" else 0 for i in data.target]
#veri uzunlugu
print(len(data))
#ilk bes veriyi gördük
print(data.head())
#verinin icindekiler
print("Data shape ", data.shape)
#data shape deger ve attribute
data.info()
#31 tane numeric deger var attribute
describe = data.describe()

"""
standardization
missing value: none
"""

# %% EDA

# Correlation
corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation Between Features")
plt.show()
#aralarıdaki ilişkiye bakabilmek için korelasyon matrisine baktım ama çok karmaşık o yüzden tresholdla daraltıcam

# 
threshold = 0.5
filtre = np.abs(corr_matrix["target"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation Between Features w Corr Threshold 0.75")
#birbiriyle ilişkili attributeler var korelasyon

"""
there some correlated features
"""

# box plot 
data_melted = pd.melt(data, id_vars = "target",
                      var_name = "features",
                      value_name = "value")

plt.figure()
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()

"""
standardization-normalization
"""
#pozitif skewness
# pair plot 
sns.pairplot(data[corr_features], diag_kind = "kde", markers = "+",hue = "target")
plt.show()

"""
skewness
"""

# %% outlier
#outlier tespiti
y = data.target
x = data.drop(["target"],axis = 1)
columns = x.columns.tolist()

clf = LocalOutlierFactor()
y_pred = clf.fit_predict(x)
X_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()
outlier_score["score"] = X_score

# threshold
threshold = -2.5
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()


plt.figure()
#2 den büyük değerleri görmek istemiyorum outlier olduğu için
plt.scatter(x.iloc[outlier_index,0], x.iloc[outlier_index,1],color = "blue", s = 50, label = "Outliers")
#data frame i aldığımız için index.tolist
plt.scatter(x.iloc[:,0], x.iloc[:,1], color = "k", s = 3, label = "Data Points")
#eğer radius büyükse outlier olmaya yakın küçükse outlier olmaya uzak
radius = (X_score.max() - X_score)/(X_score.max() - X_score.min())
outlier_score["radius"] = radius
#0-1 columnlarını kullanıyorum facecolors içi boş 
plt.scatter(x.iloc[:,0], x.iloc[:,1], s = 1000*radius, edgecolors = "r",facecolors = "none", label = "Outlier Scores")
#labellar görünebilmesi için
plt.legend()
plt.show()

# drop outliers

#outlierları çıkarıyorum
x = x.drop(outlier_index)
#y yi aray yaptık
y = y.drop(outlier_index).values

# %% Train test split
test_size = 0.3
"""
y_train class variable,xtrainde 394 sample,xtest 170 tane sample şimdi sklearnde shuffle diye bir parametre var
default olarak tanumlı ve karıştırıyor random statei tanımlamamızın sebebi herhangi bir parametrede değişiklik yaparsam
bu değişiklik random statedeki karışmadan dolayı değişmesini kontrol etmek için tanımlı 
"""
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = test_size, random_state = 42)

# %% 

scaler = StandardScaler()
"""
arada çok uçurum olan değerler için attributeler için scale etmem gerekli standardize çünkü
eğer standardize yapmazsam o attributeu silebilir ve bu bizim için anlamlı bir feature mu bilemeyiz
train verisini transform edip fitliyorum ama test verisini fit yapmıyoruz çünkü train verisine 
göre transform gerekli
"""
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_df = pd.DataFrame(X_train, columns = columns)
X_train_df_describe = X_train_df.describe()
X_train_df["target"] = Y_train
# box plot 
data_melted = pd.melt(X_train_df, id_vars = "target",
                      var_name = "features",
                      value_name = "value")

plt.figure()
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()


# pair plot 
sns.pairplot(X_train_df[corr_features], diag_kind = "kde", markers = "+",hue = "target")
plt.show()


# %% Basic KNN Method

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(Y_test, y_pred)
acc = accuracy_score(Y_test, y_pred)
score = knn.score(X_test, Y_test)
print("Score: ",score)
print("CM: ",cm)
print("Basic KNN Acc: ",acc)

# %% choose best parameters

def KNN_Best_Params(x_train, x_test, y_train, y_test):
    
    k_range = list(range(1,31))
    weight_options = ["uniform","distance"]
    print()
    param_grid = dict(n_neighbors = k_range, weights = weight_options)
    
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv = 10, scoring = "accuracy")
    grid.fit(x_train, y_train)
    
    print("Best training score: {} with parameters: {}".format(grid.best_score_, grid.best_params_))
    print()
    
    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train, y_train)
    
    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    print("Test Score: {}, Train Score: {}".format(acc_test, acc_train))
    print()
    print("CM Test: ",cm_test)
    print("CM Train: ",cm_train)
    
    return grid
    
    
grid = KNN_Best_Params(X_train, X_test, Y_train, Y_test)

# %% PCA
"""
yeniden scale yapıyoruz pca için bunun sebebi pca unsupervised learning olan bir algoritma
herhangi bir class labelına ihtiyaç duymuyor bu yüzden sadece x train verisi değil tüm x verilerini
kullanıcazv veri ayrılmıcak yani
"""
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components = 2)
#normalde 30 tane feature var bunu 2 ye düşürüyoruz
pca.fit(x_scaled)
X_reduced_pca = pca.transform(x_scaled)
pca_data = pd.DataFrame(X_reduced_pca, columns = ["p1","p2"])
pca_data["target"] = y
sns.scatterplot(x = "p1", y = "p2", hue = "target", data = pca_data)
#buraya kadar sadece 30 u 2 ye düşürdük ve görselleştirme yaptık
plt.title("PCA: p1 vs p2")


X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(X_reduced_pca, y, test_size = test_size, random_state = 42)

grid_pca = KNN_Best_Params(X_train_pca, X_test_pca, Y_train_pca, Y_test_pca)

# visualize 
cmap_light = ListedColormap(['orange',  'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

h = .05 # step size in the mesh
X = X_reduced_pca
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
"""
plot oluşturduktan sonra bu plotta x koordinatı ve y koordinatında 0.05 lik gridlere böl  
bunları topla hepsini ve ravel metoduyla düzleştir xx ve yy den sonra 188550,2lik matris elde
ettim ,2 x ve y ekseni için predict metoduyla her noktayı knn e uygulayıp tahmin edicem
bu bir tabloyu bölüp sınıflandırma yaptığımız işlemin yanlış olup olmadığını anlayabilcez
"""

Z = grid_pca.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
#iki classın renklerini belirliyor
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
#eksenlerin boyutlarını ayarlıyorum
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classification (k = %i, weights = '%s')"
          % (len(np.unique(y)),grid_pca.best_estimator_.n_neighbors, grid_pca.best_estimator_.weights))
"""
yukarıdaki sarıların bazıları yanlış sınıflandırılmış yeri farklı olmasına rağmen sarı classa
ve ortadaki mavilerde yine yanlış sınıflandırılmış
"""

#%% NCA
"""
rastgele bir  uzaklık metriği belirlemek yerine doğrusal dönüşümü bularak bu metriği NCA
kendi öğreniyor.leave one out sınıflandırma algoritması belirli bir  mesafe ölçütü kullanarak
KNNi başka tek bir noktayı predict etmeye çalıştığı yöntemdir

fit işlemi yaparken diagnosis verisine ihtiyaç duyar M-B değerli attirubute
pca ye göre daha az bir iç içe verilerin geçmesi söz konusu
uzaktaki farklı sınıflandırılmış noktalar için algoritmada yanlış bişeyler çıkabilir ama outlier 
gibi
test score 99 train 100 gayet iyi bir sınıflandırma oluyor 1 tane hata var sadece
"""
nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state = 42)
nca.fit(x_scaled, y)
X_reduced_nca = nca.transform(x_scaled)
nca_data = pd.DataFrame(X_reduced_nca, columns = ["p1","p2"])
nca_data["target"] = y
sns.scatterplot(x = "p1",  y = "p2", hue = "target", data = nca_data)
plt.title("NCA: p1 vs p2")

X_train_nca, X_test_nca, Y_train_nca, Y_test_nca = train_test_split(X_reduced_nca, y, test_size = test_size, random_state = 42)

grid_nca = KNN_Best_Params(X_train_nca, X_test_nca, Y_train_nca, Y_test_nca)

# visualize 
cmap_light = ListedColormap(['orange',  'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

h = .2 # step size in the mesh
X = X_reduced_nca
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = grid_nca.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classification (k = %i, weights = '%s')"
          % (len(np.unique(y)),grid_nca.best_estimator_.n_neighbors, grid_nca.best_estimator_.weights))

# %% find wrong decision
knn = KNeighborsClassifier(**grid_nca.best_params_)
knn.fit(X_train_nca,Y_train_nca)
y_pred_nca = knn.predict(X_test_nca)
acc_test_nca = accuracy_score(y_pred_nca,Y_test_nca)
knn.score(X_test_nca,Y_test_nca)

test_data = pd.DataFrame()
test_data["X_test_nca_p1"] = X_test_nca[:,0]
test_data["X_test_nca_p2"] = X_test_nca[:,1]
test_data["y_pred_nca"] = y_pred_nca
test_data["Y_test_nca"] = Y_test_nca

plt.figure()
sns.scatterplot(x="X_test_nca_p1", y="X_test_nca_p2", hue="Y_test_nca",data=test_data)

diff = np.where(y_pred_nca!=Y_test_nca)[0]
plt.scatter(test_data.iloc[diff,0],test_data.iloc[diff,1],label = "Wrong Classified",alpha = 0.2,color = "red",s = 1000)
#n_neighbour neden 1 çıktı mavi noktalara başarı elde ettik aradakiler kendine nasıl area çizdi
"""
train veri setinden olduğu için en yakın noktalara baktı meshgrid yaptığımı için sınıflandırabildik
ama yukarıdaki sarı nokta test verisi olduğu için sınıflandırma düzgün olmadı 
"""




