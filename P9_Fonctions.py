import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
from tabulate import tabulate
from scipy import stats
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering

# Fonction pour afficher des messages plus esthétiques
def Text_message(message):
    max_line_length = 110  # Longueur maximale d'une ligne avant de sauter à la ligne suivante
    lines = []
    current_line = ""

    for mot in message.split():
        if len(current_line + mot) <= max_line_length:
            current_line += mot + " "
        else:
            lines.append(current_line.strip())
            current_line = mot + " "
    
    # Ajoute la dernière ligne restante
    lines.append(current_line.strip())
    
    formatted_lines = "<br>".join(lines)
    styled_message = '<div style="text-align: left;"><span style="font-weight: bold; font-style: italic; font-family: Times New Roman;color: teal; font-size: 12pt; font-family: Arial;">{}</span></div>'.format(formatted_lines)
    display(HTML(styled_message))
    

# Fonction pour afficher des messages plus esthétiques
def Titre_message(message):
    max_line_length = 90  # Longueur maximale d'une ligne avant de sauter à la ligne suivante
    lines = []
    current_line = ""

    for mot in message.split():
        if len(current_line + mot) <= max_line_length:
            current_line += mot + " "
        else:
            lines.append(current_line.strip())
            current_line = mot + " "
    
    # Ajoute la dernière ligne restante
    lines.append(current_line.strip())
    
    formatted_lines = "<br>".join(lines)
    styled_message = '<div style="text-align: center;"><span style="font-weight: bold; font-style: italic; font-family: Times New Roman;color: firebrick; font-size: 14pt; font-family: Arial;">{}</span>'.format(message)
    display(HTML(styled_message))
    
    
# Lecture des informations du fichier
def infos_DF(DF):
    message = "Information du Fichier"
    Titre_message(message)
    display(DF.head(3))
    message = "le fichier contient {:.0f} lignes et {} colonnes".format(DF.shape[0],DF.shape[1])
    Text_message(message)
    print('')
    DF.info()
      # Vérification valeurs manquantes
    message = "Valeurs manquantes"
    Text_message(message)
    # Vérification valeurs uniques
    print(DF.isnull().sum())
    message = "Valeurs uniques"
    Text_message(message)
    print(DF.nunique())
    
    
# Informations sur les pays présents ou non sur la pemière table
def Pays_absents(DF1, DF2,L='Fr'):
    message = 'Concordance des Pays entre les tables'
    if L == 'Fr':
        Pays = "Pays"
        Pays2 = "Pays"
    elif L == 'En':
        Pays = "Pays"
        Pays2 = "Country_EN"
    
    pays_DF2 = DF2[Pays2].unique()
    pays_presents1 = DF1[Pays].isin(pays_DF2)
    pays_absents1 = DF1[~pays_presents1]    
    display(DF1.head(2))
    message = 'Pays absent:'
    Text_message(message)
    print(pays_absents1[Pays].unique())    


# Vérification du nombre de valeurs = 0 et du nombre de Nan    
def Verif_col(DF):
    # Créez une liste pour stocker les résultats
    resultats = []
    message = "Liste des valeurs = 0 et des NaN"
    Titre_message(message)
    # Parcourez les colonnes du DataFrame
    for colonne in DF.columns:
        # Comptez le nombre de zéros
        nb_valeurs_zero = (DF[colonne] == 0).sum()
        # Comptez le nombre de valeurs NaN
        nb_nan = DF[colonne].isna().sum()
        if nb_valeurs_zero >= (len(DF)*0.3):
            retirer = "A Suprimer"
        elif nb_valeurs_zero >= (len(DF)*0.01) or nb_nan != 0:
            retirer = "A Verifier"
        else:
            retirer = ""
            
        # Ajoutez les résultats à la liste
        resultats.append([colonne, nb_valeurs_zero, nb_nan,retirer])
    
    # Personnalisez le format HTML
    html_table = tabulate(resultats, headers=["Colonne", "Zéro", "NaN",""], tablefmt="html")

    styled_message = '<div style="text-align: center;"><span style="font-weight: bold; font-style: italic; font-family: Arial; color: firebrick; font-size: 18;">{}</span>'.format(html_table)
    # Affichez le tableau HTML
    display(HTML(styled_message))

    message = 'Format de la Table : {}'.format(DF.shape)
    Text_message(message)


# -----------------------------------   CLASSIFICATION ASCENDANTE HIERARCHIQUE --------------------------------------------
def CAH(DF, scaler = preprocessing.StandardScaler()): 
    global X_scaled  # Déclaration de X_scaled comme variable globale
    if scaler == 'Log':
        message = 'Prétraitement appliqué:  Transformation logarithmique'
        Text_message(message)
        DF_clean = DF.mask(DF <= 0, 0.1)
        # Effectuer la transformation logarithmique sur les données
        X_scaled = np.log1p(DF_clean)  # Utilisation de log1p pour éviter les erreurs avec les valeurs nulles ou négatives
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")        

    else: 
        message = f'Prétraitement appliqué:  {scaler}'
        Text_message(message)
        # Extraction des valeurs et des indexs
        X = DF.values
        y = DF.index
        # Centrage et réduction
        X_scaled = scaler.fit_transform(X)    
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")
        X_scaled = pd.DataFrame(X_scaled, columns=DF.columns)
        pays = DF.copy()
        pays.reset_index(inplace=True)
        pays = pays.loc[X_scaled.index, 'Pays']
        X_scaled.index = pays

    display(X_scaled.head(3))

    # Affichage du dendrogramme global
    message = 'CLASSIFICATION ASCENDANTE HIERARCHIQUE'
    Titre_message(message) 
    fig = plt.figure(figsize=(12, 3))
    sns.set_style('white')
    plt.ylabel('Distance')
    dendrogram(Z, labels=DF.index, leaf_font_size=5, color_threshold=12, orientation='top')
    plt.show()
    
    # Effectuer la CAH avec différentes coupes du dendrogramme
    silhouette_scores = []

    for num_clusters in range(2, 10):
        clusters = fcluster(Z, num_clusters, criterion='maxclust')
        silhouette_avg = silhouette_score(X_scaled, clusters)
        silhouette_scores.append(silhouette_avg)

    # Tracer le graphique du score de silhouette moyen pour chaque nombre de clusters
    message = 'Score de silhouette moyen pour chaque nombre de clusters - CAH'
    Titre_message(message) 
    fig, ax1 = plt.subplots(figsize=(12, 3))
    plt.plot(range(2, 10), silhouette_scores, marker='o', color='teal')
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Score de silhouette moyen")
    plt.grid()
    plt.show()

    # Trouver le nombre optimal de clusters qui maximise le score de silhouette moyen
    optimal_C = np.argmax(silhouette_scores) + 2  # +2 car on commence à 2 clusters
    message = f"Nombre optimal de clusters conseillé : {optimal_C}"
    Text_message(message)  

def CAH_groupes(DF, scaler = preprocessing.StandardScaler(), Nbc=5, Seuil_Outliers = 5):
    global X_scaled  # Déclaration de X_scaled comme variable globale
    if scaler == 'Log':
        message = 'Prétraitement appliqué:  Transformation logarithmique'
        Text_message(message)
        DF_clean = DF.mask(DF <= 0, 0.1)
        # Effectuer la transformation logarithmique sur les données
        X_scaled = np.log1p(DF_clean)  # Utilisation de log1p pour éviter les erreurs avec les valeurs nulles ou négatives
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")        

    else: 
        message = f'Prétraitement appliqué:  {scaler}'
        Text_message(message)
        # Extraction des valeurs et des indexs
        X = DF.values
        y = DF.index
        # Centrage et réduction
        X_scaled = scaler.fit_transform(X)    
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")
        X_scaled = pd.DataFrame(X_scaled, columns=DF.columns)
        pays = DF.copy()
        pays.reset_index(inplace=True)
        pays = pays.loc[X_scaled.index, 'Pays']
        X_scaled.index = pays    
    
    #Découpage du dendrogramme en groupes pour avoir une première idée du partitionnement
    fig = plt.figure(figsize=(12,3))
    message = f'Classification ascendante hiérarchique -:  {Nbc} clusters'
    Titre_message(message)
    plt.xlabel('distance', fontsize=14)
    dendrogram(Z, labels = X_scaled.index, p=Nbc, truncate_mode='lastp', leaf_font_size=14, orientation='left', 
               show_contracted=True)
    plt.show()
    
    # Outliers
    z_scores = stats.zscore(X_scaled)
    # Identification des indices des outliers
    outliers_indices = np.argwhere((z_scores > Seuil_Outliers) | (z_scores < -Seuil_Outliers))
    unique_outliers = np.unique(outliers_indices[:, 0])
    
    # Création du DataFrame contenant les valeurs aberrantes
    message = 'Liste de Outliers'
    Titre_message(message) 
    Outliers = DF.iloc[unique_outliers]
    display(Outliers)  
    
def CAH_Stats(DF, scaler = preprocessing.StandardScaler(), Nbc=5):
    global X_scaled  # Déclaration de X_scaled comme variable globale
    if scaler == 'Log':
        message = 'Prétraitement appliqué:  Transformation logarithmique'
        Text_message(message) 
        DF_clean = DF.mask(DF <= 0, 0.1)
        # Effectuer la transformation logarithmique sur les données
        X_scaled = np.log1p(DF_clean)  # Utilisation de log1p pour éviter les erreurs avec les valeurs nulles ou négatives
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")        

    else: 
        message = f'Prétraitement appliqué:  {scaler}'
        Text_message(message)        
        # Extraction des valeurs et des indexs
        X = DF.values
        y = DF.index
        # Centrage et réduction
        X_scaled = scaler.fit_transform(X)    
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")
        X_scaled = pd.DataFrame(X_scaled, columns=DF.columns)
        pays = DF.copy()
        pays.reset_index(inplace=True)
        pays = pays.loc[X_scaled.index, 'Pays']
        X_scaled.index = pays
        
    clusters = fcluster(Z, Nbc, criterion='maxclust')
    # Création d'un DF d'attribution des clusters par Pays
    DF_Clusters = pd.DataFrame({"Clusters": clusters, "Country": X_scaled.index})
    # Ajout des Clusters au DF
    X_scaled["cluster"] = clusters    
    # Interprétation des groupes - Afficher les statistiques des clusters
    cluster_stats = X_scaled.groupby('cluster')[X_scaled.columns].mean()
    # Affichage
    message = "Statistiques (moyennes) des clusters - CAH"
    Titre_message(message)
    display(cluster_stats)    
    
    # Créer un boxplot pour chaque colonne numérique à l'exception de 'cluster'
    num_cols = len(X_scaled.columns) - 1  # Exclure la colonne 'cluster'
    # Nombre de lignes pour les sous-graphiques
    num_rows = num_cols // 5
    if num_cols % 5:
        num_rows += 1

    message = "Tendance des indicateurs par clusters - CAH"
    Titre_message(message)    
    plt.figure(figsize=(10, 2 * num_rows))
    for i, column in enumerate(DF.columns, start=1):
        if column != 'cluster':  # Vérifier si la colonne est différente de 'cluster'
            plt.subplot(num_rows, 5, i)
            sns.boxplot(x='cluster', y=column, data=X_scaled, palette = 'crest')
            plt.title(column)
            plt.ylabel('')

    plt.tight_layout()
    plt.show()    

    message = "Imputation des clusters par pays - CAH"
    Titre_message(message)
    display(X_scaled.head(3))

    # Affichage des pays par Clusters
    message = "Répartition des pays par cluster - CAH"
    Titre_message(message)
    for i in range(1, Nbc + 1):
        cluster_name = f'cluster{i}'
        cluster = X_scaled[X_scaled['cluster'] == i]
        message = f'Cluster {i}: Nombre de Pays {len(cluster)}'
        Text_message(message)
        print(cluster.index.unique())     
    return X_scaled

# ---------------------------------------------------   K-Means ----------------------------------------------------------
def Kmeans(DF, scaler = preprocessing.StandardScaler()):
    message = 'K-Means'
    Titre_message(message) 
    X = DF.values 
    global X_scaled  # Déclaration de X_scaled comme variable globale
    if scaler == 'Log':
        message = 'Prétraitement appliqué:  Transformation logarithmique'
        Text_message(message)
        DF_clean = DF.mask(DF <= 0, 0.1)
        # Effectuer la transformation logarithmique sur les données
        X_scaled = np.log1p(DF_clean)  # Utilisation de log1p pour éviter les erreurs avec les valeurs nulles ou négatives
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")        

        # Appliquer la méthode de clustering hiérarchique sur les données transformées
        #Z = linkage(DF_log, method='ward')  # Vous pouvez utiliser une autre méthode si nécessaire
    else: 
        message = f'Prétraitement appliqué:  {scaler}'
        Text_message(message)
        # Extraction des valeurs et des indexs
        X = DF.values
        y = DF.index
        # Centrage et réduction
        X_scaled = scaler.fit_transform(X)    
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")
        X_scaled = pd.DataFrame(X_scaled, columns=DF.columns)
        pays = DF.copy()
        pays.reset_index(inplace=True)
        pays = pays.loc[X_scaled.index, 'Pays']
        X_scaled.index = pays
    
    
    # Extraction des valeurs et des indexs
    X = X_scaled.values
    y = X_scaled.index   
    # Liste pour stocker les valeurs d'inertie et de silhouette
    inertie = []
    inertie1 = []
    silhouette = []

    # Boucle itérative pour tester 2 à 10 clusters
    for k in range(2, 10):
        kms = KMeans(n_clusters=k, random_state=0).fit(Z)
        kms1 = KMeans(n_clusters=k, random_state=0).fit(X)
        inertie.append(kms.inertia_)
        inertie1.append(kms1.inertia_)
        silhouette_avg = silhouette_score(X, kms1.labels_)
        #silhouette_avg = silhouette_score(Z, kms.labels_)
        silhouette.append(silhouette_avg)
        #silhouette.append(silhouette_avg)    
               
    # Tracé des courbes pour la méthode du coude et les scores de silhouette
    message = 'Méthode du coude et Scores de silhouette pour le choix du nombre de clusters'
    Titre_message(message)  
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(range(2, 10), inertie, marker='o', color='teal')
    ax1.set_xlabel('Nombre de clusters')
    ax1.set_ylabel('Méthode du coude : Inertie basée sur Z', color='teal')

    ax2 = ax1.twinx()
    ax2.plot(range(2, 10), silhouette, marker='o', color='firebrick')
    ax2.set_ylabel('Score de silhouette moyen', color='firebrick')
    
    ax3 = ax1.twinx()
    ax3.plot(range(2, 10), inertie1, marker='o', color='royalblue')
    ax3.set_ylabel('Méthode du coude : Inertie basée sur X', color='royalblue')    
    # Déplacement de l'axe des ordonnées pour éviter le chevauchement des courbes
    ax3.spines['left'].set_position(('outward', 60))
    # Déplacer le label et les ticks de l'axe y vers la gauche
    ax3.yaxis.set_label_position('left')
    ax3.yaxis.set_ticks_position('left')
    fig.tight_layout()
    plt.grid()
    plt.show()        
    
    
    # Nombre optimal de clusters qui maximise le score de silhouette moyen
    optimal_C = np.argmax(silhouette) + 2  # A partir de 2 clusters
    message = f"Nombre optimal de clusters conseillé: {optimal_C}"
    Text_message(message)   


def Kmeans_Centroides(DF, scaler = preprocessing.StandardScaler(), Nbc=5, Seuil_Outliers=3):
    global X_scaled  # Déclaration de X_scaled comme variable globale
    if scaler == 'Log':
        message = 'Prétraitement appliqué:  Transformation logarithmique'
        Text_message(message)
        DF_clean = DF.mask(DF <= 0, 0.1)
        # Effectuer la transformation logarithmique sur les données
        X_scaled = np.log1p(DF_clean)  # Utilisation de log1p pour éviter les erreurs avec les valeurs nulles ou négatives
        X_norm = X_scaled
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")        

    else:
        message = f'Prétraitement appliqué:  {scaler}'
        Text_message(message)
        # Extraction des valeurs et des indexs
        X = DF.values
        y = DF.index
        # Centrage et réduction
        X_scaled = scaler.fit_transform(X) 
        X_norm = X_scaled
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")
        X_scaled = pd.DataFrame(X_scaled, columns=DF.columns)
        pays = DF.copy()
        pays.reset_index(inplace=True)
        pays = pays.loc[X_scaled.index, 'Pays']
        X_scaled.index = pays
        
    # Appliquer KMeans avec le nombre de clusters choisis
    kmeans = KMeans(n_clusters=Nbc, random_state=0)
    kmeans.fit(X_scaled)
    # Ajouter les labels de cluster au DataFrame
    X_scaled['cluster'] = kmeans.labels_
    
    # Affichage du nuage de points (individus) en cluster avec les centoîdes
    fig = plt.figure(figsize=(12,4))
    model = KMeans(n_clusters=Nbc)
    model.fit(Z)
    model.predict(Z)
    plt.scatter(Z[:,0], Z[:,1],c=model.predict(Z), marker='o', s=30)
    plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], marker='^', s=150, edgecolor='black',c='firebrick')
    plt.grid()
    plt.show()

    labels_unique, labels_counts = np.unique(kmeans.labels_, return_counts=True)
    # Affichage des étiquettes uniques avec leurs fréquences
    for label, count in zip(labels_unique, labels_counts):
        Text_message(f'Cluster {label}: {count} Pays')
       
    
    # Outliers
    z_scores = stats.zscore(X_scaled)
    # Identification des indices des outliers
    outliers_indices = np.argwhere((z_scores > Seuil_Outliers) | (z_scores < -Seuil_Outliers))
    unique_outliers = np.unique(outliers_indices[:, 0])
    
    # Création du DataFrame contenant les valeurs aberrantes
    message = 'Liste de Outliers'
    Titre_message(message) 
    Outliers = DF.iloc[unique_outliers]
    display(Outliers)   


def Kmeans_Stats(DF, scaler = preprocessing.StandardScaler(), Nbc=5):
    global X_scaled  # Déclaration de X_scaled comme variable globale
    
    if scaler == 'Log':
        message = 'Prétraitement appliqué:  Transformation logarithmique'
        Text_message(message)
        DF_clean = DF.mask(DF <= 0, 0.1)
        # Effectuer la transformation logarithmique sur les données
        X_scaled = np.log1p(DF_clean)  # Utilisation de log1p pour éviter les erreurs avec les valeurs nulles ou négatives
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")        

    else: 
        message = f'Prétraitement appliqué:  {scaler}'
        Text_message(message)        
        # Extraction des valeurs et des indexs
        X = DF.values
        y = DF.index
        # Centrage et réduction
        X_scaled = scaler.fit_transform(X)    
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")
        X_scaled = pd.DataFrame(X_scaled, columns=DF.columns)
        pays = DF.copy()
        pays.reset_index(inplace=True)
        pays = pays.loc[X_scaled.index, 'Pays']
        X_scaled.index = pays
        
    # Appliquer KMeans avec le nombre de clusters optimal trouvé (supposons que c'est 3 pour cet exemple)
    kmeans = KMeans(n_clusters=Nbc, random_state=0)
    kmeans.fit(X_scaled)
    # Ajouter les labels de cluster au DataFrame
    X_scaled['cluster'] = kmeans.labels_
    # Interprétation des groupes
    # Afficher les statistiques des clusters
    cluster_stats = X_scaled.groupby('cluster')[X_scaled.columns].mean()
    # Affichage
    message = "Statistiques (moyennes) des clusters - Kmeans"
    Titre_message(message)  
    display(cluster_stats)
    
    # Créer un boxplot pour chaque colonne numérique à l'exception de 'cluster'
    num_cols = len(X_scaled.columns) - 1  # Exclure la colonne 'cluster'
    # Nombre de lignes pour les sous-graphiques
    num_rows = num_cols // 5
    if num_cols % 5:
        num_rows += 1

    message = "Tendance des indicateurs par clusters - Kmeans"
    Titre_message(message)    
    plt.figure(figsize=(10, 2 * num_rows))
    for i, column in enumerate(DF.columns, start=1):
        if column != 'cluster':  # Vérifier si la colonne est différente de 'cluster'
            plt.subplot(num_rows, 5, i)
            sns.boxplot(x='cluster', y=column, data=X_scaled, palette = 'flare')
            plt.title(column)
            plt.ylabel('')

    plt.tight_layout()
    plt.show()    
    
    message = "Imputation des clusters par pays - KMeans"
    Titre_message(message)
    display(X_scaled.head(3))

    message = "Répartition des pays par cluster - KMeans"
    Titre_message(message)
    for i in range(0, Nbc):
        cluster_name = f'cluster{i}'
        cluster = X_scaled[X_scaled['cluster'] == i]
        message = f'Cluster {i}: Nombre de Pays {len(cluster)}'
        Text_message(message)
        print(cluster.index.unique())    

    return X_scaled


def correlation_graph(pca, x_y, features, palette="rocket", legend_fontsize=12, label_fontsize=10, Indicsize=8, arrow_alpha=0.8):
    
     # Création d'un dictionnaire de couleurs
    column_colors = {}
    column_labels = {}
    palette = sns.color_palette(palette, len(features)) # Attribution de la palette demandée

    for i in range(pca.components_.shape[1]):
        color = palette[i]  # Utilisation de la palette de couleurs
        column_colors[i] = color
        column_labels[i] = features[i]
    
    # Extrait x et y
    x, y = x_y

    # Taille de l'image (en inches)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Pour chaque composante :
    for i in range(0, pca.components_.shape[1]):
        # Les flèches
        ax.arrow(0,
                 0,  # Start the arrow at the origin
                 pca.components_[x, i],  # 0 for PC1
                 pca.components_[y, i],  # 1 for PC2
                 head_width=0.03,
                 head_length=0.03, 
                 width=0.02, fc=column_colors[i], ec='white', alpha=arrow_alpha)

        # Les labels
        plt.text(pca.components_[x, i] + 0.03,
                 pca.components_[y, i] + 0.00,
                 features[i], fontsize=Indicsize, fontweight='bold')

    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x + 1, round(100 * pca.explained_variance_ratio_[x], 1)))
    plt.ylabel('F{} ({}%)'.format(y + 1, round(100 * pca.explained_variance_ratio_[y], 1)))

    # Le cercle
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an), color='firebrick')  # Ajoutez un cercle unitaire pour l'échelle
    
    # Créez une légende avec les noms des variables à la place des indices
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=features[i], markersize=10, 
                          markerfacecolor=color) for i, color in enumerate(column_colors.values())]

    # Placez la légende en dehors du graphique (en haut à droite)
    legend = plt.legend(handles=handles, title='Composantes', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.setp(legend.get_title(), fontsize=legend_fontsize, fontweight='bold')  # Ajustez la taille de police de la légende
    
    # Ajustez la taille de police des labels dans la légende
    for text in legend.get_texts():
        text.set_fontsize(label_fontsize)
           
    plt.title("Projection des individus (sur F{} et F{})".format(x+1, y+1) , y=1.02, 
              fontdict={'size': 16, 'weight': 'bold', 'style':'italic', 'color': 'firebrick'})
    plt.axis('equal')
    plt.show(block=False)
    
    
def Plans_Factoriels(X_projected, x_y, pca=None, labels = None, clusters=None, alpha=1, figsize=[12,8], marker="."
                     , palette='viridis'):
    
    # Transforme X_projected en np.array
    X_ = np.array(X_projected)

    # On définit la forme de la figure si elle n'a pas été donnée
    if not figsize: 
        figsize = (10,8)

    # On gère les labels
    if  labels is None : 
        labels = []
    try : 
        len(labels)
    except Exception as e : 
        raise e

    # on définit x et y 
    x, y = x_y       
        
    # On vérifie la variable axis 
    if not len(x_y) ==2 : 
        raise AttributeError("2 axes sont demandées")   
    if max(x_y )>= X_.shape[1] : 
        raise AttributeError("la variable axis n'est pas bonne")
    
    
    # Initialisation de la figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # On vérifie s'il y a des clusters ou non
    c = None if clusters is None else clusters 
    
    # Les points
    # plt.scatter(X_[:, x], X_[:, y], alpha=alpha, c=c, cmap="Set1", marker=marker)
    sns.scatterplot(data=None, x=X_[:, x], y=X_[:, y], hue=c, palette=palette, marker=marker,alpha=1)

    # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe
    if pca:
        v1 = str(round(100 * pca.explained_variance_ratio_[x])) + " %"
        v2 = str(round(100 * pca.explained_variance_ratio_[y])) + " %"
    else:
        v1 = v2 = ''    
    
    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f'F{x+1} {v1}')
    ax.set_ylabel(f'F{y+1} {v2}')

    # Valeur x max et y max
    x_max = np.abs(X_[:, x]).max() * 1.1
    y_max = np.abs(X_[:, y]).max() * 1.1

    # On borne x et y
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom=-y_max, top=y_max)

    # Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color='grey', linestyle='dashed', alpha=alpha)
    plt.plot([0, 0], [-y_max, y_max], color='grey', linestyle='dashed', alpha=alpha)
    
    # Affichage des labels des points
    if len(labels):
        for i, (_x, _y) in enumerate(X_[:, [x, y]]):
            plt.text(_x, _y + 0.05, labels[i], fontsize='12', ha='left', va='center')

    # Titre et display
    plt.title(f"Projection des individus (sur F{x+1} et F{y+1})", color='teal', fontsize=16, fontweight='bold')
    plt.show()
    
        

    
def CAH_Clust(DF, scaler = preprocessing.StandardScaler(), Nbc=5, Seuil_Outliers = 5, figsize=[6,3]):
    global X_scaled  # Déclaration de X_scaled comme variable globale
    if scaler == 'Log':
        message = 'Prétraitement appliqué:  Transformation logarithmique'
        Text_message(message)
        DF_clean = DF.mask(DF <= 0, 0.1)
        # Effectuer la transformation logarithmique sur les données
        X_scaled = np.log1p(DF_clean)  # Utilisation de log1p pour éviter les erreurs avec les valeurs nulles ou négatives
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")        

    else: 
        message = f'Prétraitement appliqué:  {scaler}'
        Text_message(message)
        # Extraction des valeurs et des indexs
        X = DF.values
        y = DF.index
        # Centrage et réduction
        X_scaled = scaler.fit_transform(X)    
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")
        X_scaled = pd.DataFrame(X_scaled, columns=DF.columns)
        pays = DF.copy()
        pays.reset_index(inplace=True)
        pays = pays.loc[X_scaled.index, 'Pays']
        X_scaled.index = pays    
    
    #Découpage du dendrogramme en groupes pour avoir une première idée du partitionnement
    fig = plt.figure(figsize=figsize)
    message = f'Classification ascendante hiérarchique -:  {Nbc} clusters'
    Titre_message(message)
    plt.xlabel('distance', fontsize=14)
    dendrogram(Z, labels = X_scaled.index, p=Nbc, truncate_mode='lastp', leaf_font_size=14, orientation='left', 
               show_contracted=True)
    plt.show()
    
    # Outliers
    z_scores = stats.zscore(X_scaled)
    # Identification des indices des outliers
    outliers_indices = np.argwhere((z_scores > Seuil_Outliers) | (z_scores < -Seuil_Outliers))
    unique_outliers = np.unique(outliers_indices[:, 0])
    
def Kmeans_Clust(DF, scaler = preprocessing.StandardScaler(), Nbc=5, Seuil_Outliers=3, figsize=[6,3]):
    global X_scaled  # Déclaration de X_scaled comme variable globale
    if scaler == 'Log':
        message = 'Prétraitement appliqué:  Transformation logarithmique'
        Text_message(message)
        DF_clean = DF.mask(DF <= 0, 0.1)
        # Effectuer la transformation logarithmique sur les données
        X_scaled = np.log1p(DF_clean)  # Utilisation de log1p pour éviter les erreurs avec les valeurs nulles ou négatives
        X_norm = X_scaled
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")        

    else:
        message = f'Prétraitement appliqué:  {scaler}'
        Text_message(message)
        # Extraction des valeurs et des indexs
        X = DF.values
        y = DF.index
        # Centrage et réduction
        X_scaled = scaler.fit_transform(X) 
        X_norm = X_scaled
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")
        X_scaled = pd.DataFrame(X_scaled, columns=DF.columns)
        pays = DF.copy()
        pays.reset_index(inplace=True)
        pays = pays.loc[X_scaled.index, 'Pays']
        X_scaled.index = pays
        
    # Appliquer KMeans avec le nombre de clusters choisis
    kmeans = KMeans(n_clusters=Nbc, random_state=0)
    kmeans.fit(X_scaled)
    # Ajouter les labels de cluster au DataFrame
    X_scaled['cluster'] = kmeans.labels_
    
    # Affichage du nuage de points (individus) en cluster avec les centoîdes
    fig = plt.figure(figsize=figsize)
    model = KMeans(n_clusters=Nbc)
    model.fit(Z)
    model.predict(Z)
    plt.scatter(Z[:,0], Z[:,1],c=model.predict(Z))
    plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], marker='^',c='c')
    plt.grid()
    plt.show()
    # Etiquettes et fréquences
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    # Affichage
    message = ' '.join([f'Cluster {label}: {count} Pays -- ' for label, count in zip(labels, counts)])
    Text_message(message)
    
    
    
def CAH_Clust(DF, scaler = preprocessing.StandardScaler(), Nbc=5, Seuil_Outliers = 5, figsize=[6,3]):
    global X_scaled  # Déclaration de X_scaled comme variable globale
    if scaler == 'Log':
        message = 'Prétraitement appliqué:  Transformation logarithmique'
        Text_message(message)
        DF_clean = DF.mask(DF <= 0, 0.1)
        # Effectuer la transformation logarithmique sur les données
        X_scaled = np.log1p(DF_clean)  # Utilisation de log1p pour éviter les erreurs avec les valeurs nulles ou négatives
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")        

    else: 
        message = f'Prétraitement appliqué:  {scaler}'
        Text_message(message)
        # Extraction des valeurs et des indexs
        X = DF.values
        y = DF.index
        # Centrage et réduction
        X_scaled = scaler.fit_transform(X)    
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")
        X_scaled = pd.DataFrame(X_scaled, columns=DF.columns)
        pays = DF.copy()
        pays.reset_index(inplace=True)
        pays = pays.loc[X_scaled.index, 'Pays']
        X_scaled.index = pays    
    
    #Découpage du dendrogramme en groupes pour avoir une première idée du partitionnement
    fig = plt.figure(figsize=figsize)
    message = f'Classification ascendante hiérarchique -:  {Nbc} clusters'
    Titre_message(message)
    plt.xlabel('distance', fontsize=14)
    dendrogram(Z, labels = X_scaled.index, p=Nbc, truncate_mode='lastp', leaf_font_size=14, orientation='left', 
               show_contracted=True)
    plt.show()
    
    # Outliers
    z_scores = stats.zscore(X_scaled)
    # Identification des indices des outliers
    outliers_indices = np.argwhere((z_scores > Seuil_Outliers) | (z_scores < -Seuil_Outliers))
    unique_outliers = np.unique(outliers_indices[:, 0])
    
    # ---------------- K-means ----------------------    
    # Appliquer KMeans avec le nombre de clusters choisis
    kmeans = KMeans(n_clusters=Nbc, random_state=0)
    kmeans.fit(X_scaled)
    # Ajouter les labels de cluster au DataFrame
    X_scaled['cluster'] = kmeans.labels_
    
    # Affichage du nuage de points (individus) en cluster avec les centoîdes
    fig = plt.figure(figsize=figsize)
    model = KMeans(n_clusters=Nbc)
    model.fit(Z)
    model.predict(Z)
    plt.scatter(Z[:,0], Z[:,1],c=model.predict(Z))
    plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], marker='^',c='c')
    plt.grid()
    plt.show()
    # Etiquettes et fréquences
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    # Affichage
    message = ' '.join([f'Cluster {label}: {count} Pays -- ' for label, count in zip(labels, counts)])
    Text_message(message)    
    

def Clusterings(DF, scaler = preprocessing.StandardScaler(), Nbc=5, Seuil_Outliers = 5, figsize=[10,2]):
    global X_scaled  # Déclaration de X_scaled comme variable globale
    if scaler == 'Log':
        message = 'Prétraitement appliqué:  Transformation logarithmique'
        Text_message(message)
        DF_clean = DF.mask(DF <= 0, 0.1)
        # Effectuer la transformation logarithmique sur les données
        X_scaled = np.log1p(DF_clean)  # Utilisation de log1p pour éviter les erreurs avec les valeurs nulles ou négatives
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")        

    else: 
        message = f'Prétraitement appliqué:  {scaler}'
        Text_message(message)
        # Extraction des valeurs et des indexs
        X = DF.values
        y = DF.index
        # Centrage et réduction
        X_scaled = scaler.fit_transform(X)    
        #Clustering hiérarchique: 
        Z = linkage(X_scaled, method="ward", metric="euclidean")
        X_scaled = pd.DataFrame(X_scaled, columns=DF.columns)
        pays = DF.copy()
        pays.reset_index(inplace=True)
        pays = pays.loc[X_scaled.index, 'Pays']
        X_scaled.index = pays     

    # Affichage du nuage de points (individus) en cluster avec les centoîdes
    fig = plt.figure(figsize=figsize)
    plt.subplot(1, 3, 1)
    # ---------------- CAH ----------------------
    #Découpage du dendrogramme en groupes pour avoir une première idée du partitionnement
    plt.title('Classification ascendante hiérarchique',color='firebrick',fontsize='12',fontweight='bold')
    plt.xlabel('')
    dendrogram(Z, labels = X_scaled.index, p=Nbc, truncate_mode='lastp', leaf_font_size=14, orientation='left', 
               show_contracted=True)    
    #Memo fonction dendrogram:
    #dendrogram(Z, labels=X_scaled.index, p=Nbc, truncate_mode='lastp', leaf_font_size=14, orientation='left',
          # show_contracted=True, leaf_rotation=0, color_threshold=2, above_threshold_color='blue')
    

    plt.subplot(1, 3, 2)
    # ---------------- K-means ----------------------    
    # Appliquer KMeans avec le nombre de clusters choisis
    kmeans = KMeans(n_clusters=Nbc, random_state=0)
    kmeans.fit(X_scaled)
    # Ajouter les labels de cluster au DataFrame
    X_scaled['cluster'] = kmeans.labels_
    model = KMeans(n_clusters=Nbc)
    model.fit(Z)
    model.predict(Z)
    plt.scatter(Z[:,0], Z[:,1], marker='o', c=model.predict(Z), s=10, cmap='viridis')
    plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], marker='^', s=80, edgecolor='black', c='firebrick')
    plt.title('K-means',color='firebrick',fontsize='12',fontweight='bold')
    plt.grid()    
    
    plt.subplot(1, 3, 3)    
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    for label, count in zip(labels, counts):
        plt.text(0, 0.8 - label * 0.2, f'Cluster {label}  --> {count} Pays', ha='left', fontsize=14, color='Teal')
    plt.axis('off')  # Masquer les axes
    plt.title('Clusters K-means',color='firebrick',fontsize='12',fontweight='bold', loc='left')
    plt.tight_layout()
    plt.show()    
    