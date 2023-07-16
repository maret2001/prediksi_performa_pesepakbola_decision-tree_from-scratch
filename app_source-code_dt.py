import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy import stats
import pickle
from st_aggrid import AgGrid
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import f1_score,precision_score,recall_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Prediksi Performa Pesepakbola", page_icon="⚽")
# st.set_option("deprecation.showfileUploaderEncoding", False)

def load_scaler():
  scaler_data = "D:/My Drive/Semester 8/tugas_akhir/liga champions/skripsi/scaler.pkl"
  with open(scaler_data, 'rb') as file1:
    scaler = pickle.load(file1)
  return scaler

scaler = load_scaler()


def main():

  # --- Bagian Source Code Decision Tree Classifier tanpa parameter ---

  class Node:
      def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None):
          self.feature_idx = feature_idx
          self.threshold = threshold
          self.value = value
          self.left = left
          self.right = right

  class DT_Classifier_sklearn:
      def __init__(self):
          self.scaler = scaler
          self.root = None

      def gini_index(self, y):
          _, counts = np.unique(y, return_counts=True)
          probabilities = counts / len(y)
          gini = 1 - np.sum(probabilities**2)
          return gini

      def find_best_split(self, X, y):
          n_features = X.shape[1]
          best_gini = np.inf
          best_feature_idx = None
          best_threshold = None

          for feature_idx in range(n_features):
              thresholds = np.unique(X[:, feature_idx])
              for threshold in thresholds:
                  left_indices = X[:, feature_idx] <= threshold
                  right_indices = X[:, feature_idx] > threshold

                  gini_left = self.gini_index(y[left_indices])
                  gini_right = self.gini_index(y[right_indices])
                  gini = (len(y[left_indices]) * gini_left + len(y[right_indices]) * gini_right) / len(y)

                  if gini < best_gini:
                      best_gini = gini
                      best_feature_idx = feature_idx
                      best_threshold = threshold

          return best_feature_idx, best_threshold

      def build_tree(self, X, y):
          if len(np.unique(y)) == 1:
              return Node(value=np.unique(y)[0])

          feature_idx, threshold = self.find_best_split(X, y)
          left_indices = X[:, feature_idx] <= threshold
          right_indices = X[:, feature_idx] > threshold

          left = self.build_tree(X[left_indices], y[left_indices])
          right = self.build_tree(X[right_indices], y[right_indices])

          return Node(feature_idx=feature_idx, threshold=threshold, left=left, right=right)

      def fit(self, X, y):
          X_scaled = self.scaler.fit_transform(X)
          self.root = self.build_tree(X_scaled, y)

      def predict(self, X):
          X_scaled = self.scaler.transform(X)
          return np.array([self.traverse_tree(x, self.root) for x in X_scaled])

      def traverse_tree(self, x, node):
          if node.value is not None:
              return node.value

          if x[node.feature_idx] <= node.threshold:
              return self.traverse_tree(x, node.left)
          else:
              return self.traverse_tree(x, node.right)

  # --- Bagian Source Code Decision Tree Classifier tanpa parameter ---


  # --- Bagian Source Code Decision Tree Classifier dengan parameter ---

  class Node:
      def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None):
          self.feature_idx = feature_idx
          self.threshold = threshold
          self.value = value
          self.left = left
          self.right = right

  class DT_Classifier_source_code:
      def __init__(self, max_depth, max_leaf_nodes):
          self.max_depth = max_depth
          self.max_leaf_nodes = max_leaf_nodes
          self.scaler = scaler
          self.root = None

      def gini_index(self, y):
          _, counts = np.unique(y, return_counts=True)
          probabilities = counts / len(y)
          gini = 1 - np.sum(probabilities**2)
          return gini

      def find_best_split(self, X, y):
          n_features = X.shape[1]
          best_gini = np.inf
          best_feature_idx = None
          best_threshold = None

          for feature_idx in range(n_features):
              thresholds = np.unique(X[:, feature_idx])
              for threshold in thresholds:
                  left_indices = X[:, feature_idx] <= threshold
                  right_indices = X[:, feature_idx] > threshold

                  gini_left = self.gini_index(y[left_indices])
                  gini_right = self.gini_index(y[right_indices])
                  gini = (len(y[left_indices]) * gini_left + len(y[right_indices]) * gini_right) / len(y)

                  if gini < best_gini:
                      best_gini = gini
                      best_feature_idx = feature_idx
                      best_threshold = threshold

          return best_feature_idx, best_threshold

      def build_tree(self, X, y, depth=0):
          if depth >= self.max_depth or len(np.unique(y)) == 1 or len(X) < self.max_leaf_nodes:
              value = np.argmax(np.bincount(y))
              return Node(value=value)

          feature_idx, threshold = self.find_best_split(X, y)
          left_indices = X[:, feature_idx] <= threshold
          right_indices = X[:, feature_idx] > threshold

          left = self.build_tree(X[left_indices], y[left_indices], depth+1)
          right = self.build_tree(X[right_indices], y[right_indices], depth+1)

          return Node(feature_idx=feature_idx, threshold=threshold, left=left, right=right)

      def fit(self, X, y):
          X_scaled = self.scaler.fit_transform(X)
          self.root = self.build_tree(X_scaled, y)

      def predict(self, X):
          X_scaled = self.scaler.transform(X)
          return np.array([self.traverse_tree(x, self.root) for x in X_scaled])

      def traverse_tree(self, x, node):
          if node.value is not None:
              return node.value

          if x[node.feature_idx] <= node.threshold:
              return self.traverse_tree(x, node.left)
          else:
              return self.traverse_tree(x, node.right)

  # --- Bagian Source Code Decision Tree Classifier dengan parameter ---
  
  # Header
  col1, col2, col3 = st.columns(3)

  with col1:
    st.write("")

  with col2:
    img = Image.open("png-clipart-uefa-champions-league-logo-2018-uefa-champions-league-final-uefa-europa-league-europe-2012-uefa-ch.png")
    img = img.resize((250,250))
    st.image(img, use_column_width=False)

  with col3:
    st.write("")

  # judul web
  judul = "✨ Prediksi :red[Performa] Pesepakbola Profesional (Studi Kasus Kompetisi UCL 2021/2022) ✨"
  htmlJudul = f"""**<p style='background-color:white;
                              color:black;
                              font-family:Tahoma;
                              font-size:30px;
                              border-radius:3px;
                              line-height:60px;
                              text-align:center;
                              margin-top:-10px;
                              margin-bottom:-40px;
                              padding-bottom:40px;
                              opacity:0.7'>{judul}<br></p>**"""
  st.caption(htmlJudul,unsafe_allow_html=True)
  
  # Upload Data
  st.subheader("🌐 Upload Data")
  uploaded_file = st.file_uploader(" ", type= "csv")
  if uploaded_file is not None:
    print(uploaded_file)
    try:
      data = pd.read_csv(uploaded_file)
    except Exception as e:
      print(e)

  try:
    grid_table = AgGrid(data,
                        height = 500,
                        theme='balham',
                        fit_columns_on_grid_load = True) # --> menampilkan datasetnya
    jml_baris = f"📋 Jumlah Data : {len(data)} baris dan {len(data.columns)} kolom"
    htmlbaris = f"""**<p style='background-color:white;
                                color:black;
                                font-family:Tahoma;
                                font-size:17px;
                                border-radius:3px;
                                line-height:60px;
                                text-align:right;
                                margin-top:-40px;
                                margin-bottom:-20px;
                                opacity:0.6'>{jml_baris}</style><br></p>**"""
    st.caption(htmlbaris,unsafe_allow_html=True)
  except Exception as e:
    print(e)
  
  # Sidebar
  logo = "📋"
  htmlLogo = f"""**<p style='background-color:#CAEEFB;
                            font-size:70px;
                            border-radius:3px;
                            line-height:60px;
                            text-align:center;
                            margin-top:-40px;
                            margin-bottom:-70px'>{logo}<br></p>**"""
  st.sidebar.caption(htmlLogo,unsafe_allow_html=True)
  nama = "Inputan Statistik Pemain"
  htmlNama = f"""**<p style='background-color:#CAEEFB;
                            color:black;
                            font-family:Tahoma;
                            font-size:20px;
                            border-radius:3px;
                            line-height:60px;
                            text-align:center;
                            margin-top:40px;
                            margin-bottom:-70px;
                            opacity:0.7'>{nama}<br></p>**"""
  st.sidebar.caption(htmlNama,unsafe_allow_html=True)
  st.sidebar.markdown("---")
  
  tab_preprocessing, tab_modelling, tab_prediksi = st.tabs(['Menu Preprocessing','Menu Source Code','Menu Prediksi'])
  
  with tab_preprocessing:
    # Preprocessing
    st.subheader("🔄️ Preprocessing")
    st.write("""
            1. Cleaning Data
            2. Data Transformation
            3. Drop features/kolom yang tidak dibutuhkan
            4. Encoding Data
            5. Data Standarization
            6. Splitting Data
            7. Membuat Balanced Data
            """)
    col_posisi, col_performa = st.columns(2)
    with col_posisi:
        st.markdown("Keterangan Encoding posisi pemain :")
        goalkeeper = 0; defender = 1; midfielder = 2; forward = 3
        data_posisi = {'Goalkeeper': goalkeeper,
                        'Defender': defender,
                        'Midfielder': midfielder,
                        'Forward': forward}
        posisi_pemain = pd.DataFrame(data_posisi, index=[0])
        # grid_table_2 = AgGrid(posisi_pemain,
        #                 width = 50,
        #                 theme='balham',
        #                 fit_columns_on_grid_load = True)
        st.dataframe(posisi_pemain)
      
    with col_performa:
        st.markdown("Keterangan Encoding performa pemain :")
        bad = 0; normal = 1; good = 2
        data_performa = {'Bad': bad,
                        'Normal': normal,
                        'Good': good}
        performa_pemain = pd.DataFrame(data_performa, index=[0])
        # grid_table_3 = AgGrid(performa_pemain,
        #                 width = 50,
        #                 theme='balham',
        #                 fit_columns_on_grid_load = True)
        st.dataframe(performa_pemain)
    
    test_size_btn = st.radio("Pilih Test Size untuk Splitting Data :", ("20%","25%","30%"))
    
    if test_size_btn == "20%":
      test_size = 0.2
    if test_size_btn == "25%":
      test_size = 0.25
    if test_size_btn == "30%":
      test_size = 0.3
    
    tab1, tab2, tab3 = st.tabs(["Hasil Preprocessing",
                              "Hasil Reshape Data",
                              "Hasil Normalisasi Data"])
    with tab1:
      if st.checkbox("Lihat Hasil Preprocessing :"):
        df_performa = data.drop({'red','minutes_played','yellow','club','player_name','fouls_committed', 
                                'fouls_suffered', 'pass_attempted', 'cross_attempted'}, axis=1)
        
        # menghilangkan outlier
        kolom_numerik = ['pass_accuracy', 'pass_completed', 'cross_accuracy',
              'cross_completed', 'freekicks_taken', 'match_played', 'goals',
              'assists', 'distance_covered']

        # menghapus outliers dengan meanfaatkan Z-Score
        # jika suatu baris memiliki Z-Score > 3, maka baris tersebut dihapus karena memiliki outlier
        df_performa = df_performa[(np.abs(stats.zscore(df_performa[kolom_numerik])) < 3).all(axis=1)]

        # Label Encoding
        # label encoding secara manual dengan memanfaatkan fuction map()

        le_posisi = df_performa['position'].map({'Midfielder':2, 'Defender':1, 'Forward':3, 'Goalkeeper':0})
        df_performa['position'] = le_posisi

        le_performa = df_performa['performance'].map({'Bad':0, 'Normal':1, 'Good':2})
        df_performa['performance'] = le_performa
        
        
        # imbalanced data
        X = df_performa.drop(labels='performance', axis=1).values
        y = df_performa['performance'].values
        
        sm = SMOTENC(random_state=42, categorical_features=[0]) # feature ke nol (kolom position) merupakan feature kategorikal
        X_res, y_res = sm.fit_resample(X, y)
        
        # split data
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=test_size, random_state=21, stratify=y_res)
        
        # scalling
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        grid_table_4 = AgGrid(df_performa,
                              height = 500,
                              theme = 'balham',
                              fit_columns_on_grid_load = True)
        # st.dataframe(df_performa)
        jml_baris = f"📋 Jumlah Data : {len(df_performa)} baris dan {len(df_performa.columns)} kolom"
        htmlbaris = f"""**<p style='background-color:white;
                                    color:black;
                                    font-family:Tahoma;
                                    font-size:17px;
                                    border-radius:3px;
                                    line-height:60px;
                                    text-align:right;
                                    margin-top:-40px;
                                    margin-bottom:-20px;
                                    opacity:0.6'>{jml_baris}</style><br></p>**"""
        st.caption(htmlbaris,unsafe_allow_html=True)
        
    with tab2:
      if st.checkbox("Lihat Reshape Data :"):
        col_X, col_y = st.columns(2)
        with col_X:
          st.subheader(f"X reshape : {(X_res.shape)}")
          st.dataframe(X_res)
        with col_y:
          st.subheader(f"Y reshape : {(y_res.shape)}")
          st.dataframe(y_res)
    
    with tab3:
      if st.checkbox("Lihat Splitting Data"):
        train_data = f"Train set size = {X_train.shape, y_train.shape}"
        test_data = f", Test set size = {X_test.shape, y_test.shape}"
        htmlTrain_data = f"""**<p style='color:black;
                                font-family:Tahoma;
                                font-size:20px;
                                border-radius:3px;
                                line-height:60px;
                                text-align:center;
                                margin-top:-40px;
                                margin-bottom:-70px;
                                padding-bottom:70px;
                                opacity:0.7'>{train_data}\t{test_data}<br></p>**"""
        st.caption(htmlTrain_data,unsafe_allow_html=True)

      if st.checkbox("Lihat Normalisasi Data :"):
        col_X_scaled, col_y_scaled = st.columns(2)
        with col_X_scaled:
          st.subheader("X train scaled")
          st.dataframe(X_train_scaled)
        with col_y_scaled:
          st.subheader("X test scaled")
          st.dataframe(X_test_scaled)
  
  with tab_modelling:
    # Modelling
    st.subheader("🔄️ Modelling Decision Tree Classifier")
    hyperpar_btn = st.radio("Aktifkan proses Tuning Hyperparameter?", ("Tidak","Ya"))
    
    if hyperpar_btn == "Tidak":
      st.markdown("---")
      st.subheader("📶 Evaluasi")
      if st.checkbox("Lihat Evaluasi"):
        tree = DT_Classifier_sklearn()
        tree.fit(X_train_scaled, y_train)
        pred_model = tree.predict(X_test_scaled)
        acc = accuracy_score(y_test, pred_model)
        error = 1 - acc
        prec = precision_score(pred_model, y_test, average='macro')
        rec = recall_score(pred_model, y_test, average='macro')
        f1_s = f1_score(pred_model, y_test, average='macro')
        txt_acc="Accuracy : {0:.2f}%".format(acc*100)
        txt_error="⚠️ Error rate : {0:.2f}%".format(error*100)
        txt_prec="Precision : {0:.2f}%".format(prec*100)
        txt_rec="Recall : {0:.2f}%".format(rec*100)
        txt_f1_s="F1-score : {0:.2f}%".format(f1_s*100)
        cm_model = confusion_matrix(y_test, pred_model)
        htmlstr_acc=f"""**<p style='color:green;
                                  background-color:#eded5a;
                                  font-size:30px;
                                  text-align:center;
                                  border-radius:3px;
                                  line-height:60px;
                                  padding-left:30px;
                                  opacity:0.6'>{txt_acc}</style><br></p>**"""
        htmlstr_error=f"""**<p style='color:white;
                                  background-color:red;
                                  font-size:30px;
                                  text-align:center;
                                  border-radius:3px;
                                  margin-top:-20px;
                                  line-height:60px;
                                  padding-left:30px;
                                  opacity:0.6'>{txt_error}</style><br></p>**"""
        htmlstr_prec=f"""**<p style='color:green;
                                  font-size:30px;
                                  background-color:#eded5a;
                                  border-radius:3px;
                                  line-height:60px;
                                  text-align:center;
                                  margin-top:-20px;
                                  padding-left:30px;
                                  opacity:0.6'>{txt_prec}</style><br></p>**"""
        htmlstr_rec=f"""**<p style='color:green;
                                  background-color:#eded5a;
                                  font-size:30px;
                                  text-align:center;
                                  border-radius:3px;
                                  line-height:60px;
                                  padding-left:30px;
                                  opacity:0.6'>{txt_rec}</style><br></p>**"""
        htmlstr_f1_s=f"""**<p style='color:green;
                                  background-color:#eded5a;
                                  font-size:30px;
                                  text-align:center;
                                  border-radius:3px;
                                  margin-top:-20px;
                                  line-height:60px;
                                  padding-left:30px;
                                  opacity:0.6'>{txt_f1_s}</style><br></p>**"""                                                
        tab1, tab2 = st.tabs(["Nilai Evaluasi", "Confusion Matrix"])
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
              st.markdown(htmlstr_acc,unsafe_allow_html=True)
              st.markdown(htmlstr_prec,unsafe_allow_html=True)
            st.markdown(htmlstr_error,unsafe_allow_html=True)
            with col2:
              st.markdown(htmlstr_rec,unsafe_allow_html=True)
              st.markdown(htmlstr_f1_s,unsafe_allow_html=True)
        with tab2:
            st.subheader("Confusion Matrix:")
            st.dataframe(cm_model)
      
    if hyperpar_btn == "Ya":
      col_gabungan, col_kernel = st.columns(2)
      with col_gabungan:
        max_depth = st.slider("Nilai Max Depth :", 1,11,1)

      with col_kernel:
        max_leaf_nodes = st.slider("Nilai Max Leaf Nodes :", 1,11,1)
    
      model_tree = DT_Classifier_source_code(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
      
      st.markdown("---")
      st.subheader("📶 Evaluasi")
      if st.checkbox("Lihat Evaluasi"):
        model_tree.fit(X_train_scaled, y_train)
        pred_model = model_tree.predict(X_test_scaled)
        acc = accuracy_score(y_test, pred_model)
        error = 1 - acc
        prec = precision_score(pred_model, y_test, average='macro')
        rec = recall_score(pred_model, y_test, average='macro')
        f1_s = f1_score(pred_model, y_test, average='macro')
        txt_acc="Accuracy : {0:.2f}%".format(acc*100)
        txt_error="⚠️ Error rate : {0:.2f}%".format(error*100)
        txt_prec="Precision : {0:.2f}%".format(prec*100)
        txt_rec="Recall : {0:.2f}%".format(rec*100)
        txt_f1_s="F1-score : {0:.2f}%".format(f1_s*100)
        cm_model = confusion_matrix(y_test, pred_model)
        htmlstr_acc=f"""**<p style='color:green;
                                  background-color:#eded5a;
                                  font-size:30px;
                                  text-align:center;
                                  border-radius:3px;
                                  line-height:60px;
                                  padding-left:30px;
                                  opacity:0.6'>{txt_acc}</style><br></p>**"""
        htmlstr_error=f"""**<p style='color:white;
                                  background-color:red;
                                  font-size:30px;
                                  text-align:center;
                                  border-radius:3px;
                                  margin-top:-20px;
                                  line-height:60px;
                                  padding-left:30px;
                                  opacity:0.6'>{txt_error}</style><br></p>**"""
        htmlstr_prec=f"""**<p style='color:green;
                                  font-size:30px;
                                  background-color:#eded5a;
                                  border-radius:3px;
                                  line-height:60px;
                                  text-align:center;
                                  margin-top:-20px;
                                  padding-left:30px;
                                  opacity:0.6'>{txt_prec}</style><br></p>**"""
        htmlstr_rec=f"""**<p style='color:green;
                                  background-color:#eded5a;
                                  font-size:30px;
                                  text-align:center;
                                  border-radius:3px;
                                  line-height:60px;
                                  padding-left:30px;
                                  opacity:0.6'>{txt_rec}</style><br></p>**"""
        htmlstr_f1_s=f"""**<p style='color:green;
                                  background-color:#eded5a;
                                  font-size:30px;
                                  text-align:center;
                                  border-radius:3px;
                                  margin-top:-20px;
                                  line-height:60px;
                                  padding-left:30px;
                                  opacity:0.6'>{txt_f1_s}</style><br></p>**"""                                                
        tab1, tab2 = st.tabs(["Nilai Evaluasi", "Confusion Matrix"])
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
              st.markdown(htmlstr_acc,unsafe_allow_html=True)
              st.markdown(htmlstr_prec,unsafe_allow_html=True)
            st.markdown(htmlstr_error,unsafe_allow_html=True)
            with col2:
              st.markdown(htmlstr_rec,unsafe_allow_html=True)
              st.markdown(htmlstr_f1_s,unsafe_allow_html=True)
        with tab2:
            st.subheader("Confusion Matrix:")
            st.dataframe(cm_model)
  
  with tab_prediksi:
    # Prediksi
    st.subheader("#️⃣ Prediksi")
    # daftar widgetnya
    posisi = st.sidebar.selectbox('**Position :**', ('Goalkeeper','Defender','Midfielder','Forward'))
    pass_acc = st.sidebar.number_input("**Passing Accuracy (%):**", 0,100,1)
    pass_comp =st.sidebar.number_input("**Passing Complete :**", 0,1000,1)
    cross_acc = st.sidebar.number_input("**Crossing Accuracy (%):**", 0,100,1)
    cross_comp = st.sidebar.number_input("**Crossing Complete :**", 0,1000,1)
    freekicks_taken = st.sidebar.number_input("**Freekicks Taken :**", 0,1000,1)
    match_played = st.sidebar.number_input("**Match Played :**", 0,1000,1)
    jumlah_gol = st.sidebar.number_input("**Total Goals :**", 0,1000,1)
    jumlah_asis = st.sidebar.number_input("**Total Assist :**", 0,1000,1)
    dis_covered = st.sidebar.number_input("**Distance Covered :**", 0,1000,1)
    
    if st.sidebar.button("**Lihat Prediksi**"):
      # 1. Preprocess
      data_baru = {'posisi' : posisi,
                  'pass_acc' : pass_acc, 
                  'pass_comp' : pass_comp, 
                  'cross_acc' : cross_acc, 
                  'cross_comp' : cross_comp, 
                  'freekicks_taken' : freekicks_taken, 
                  'match_played' : match_played, 
                  'jumlah_gol' : jumlah_gol,
                  'jumlah_asis' : jumlah_asis, 
                  'dis_covered' : dis_covered}
      
      fitur = pd.DataFrame(data_baru, index=[0])
      col_posisi, col_performa = st.columns(2)
      with col_posisi:
        st.markdown("Keterangan Encoding posisi pemain :")
        goalkeeper = 0; defender = 1; midfielder = 2; forward = 3
        data_posisi = {'Goalkeeper': goalkeeper,
                        'Defender': defender,
                        'Midfielder': midfielder,
                        'Forward': forward}
        posisi_pemain = pd.DataFrame(data_posisi, index=[0])
        st.dataframe(posisi_pemain)
      
      with col_performa:
        st.markdown("Keterangan Encoding performa pemain :")
        bad = 0; normal = 1; good = 2
        data_performa = {'Bad': bad,
                        'Normal': normal,
                        'Good': good}
        performa_pemain = pd.DataFrame(data_performa, index=[0])
        st.dataframe(performa_pemain)
        
      st.markdown("Sebelum Encoding :")
      st.dataframe(fitur)
      le_posisi = fitur['posisi'].map({'Midfielder':2, 'Defender':1, 'Forward':3, 'Goalkeeper':0})
      fitur['posisi'] = le_posisi
      st.markdown("Setelah Encoding :")
      st.dataframe(fitur)
      st.caption("**_Catatan: 0 = Goalkeeper, 1 = Defender, 2 = Midfielder, 3 = Forward_**")
      fitur = scaler.transform(fitur)
      # 2. Source Code
      class Node:
        def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None):
            self.feature_idx = feature_idx
            self.threshold = threshold
            self.value = value
            self.left = left
            self.right = right

      class DT_Classifier_source_code:
          def __init__(self, max_depth, max_leaf_nodes):
              self.max_depth = max_depth
              self.max_leaf_nodes = max_leaf_nodes
              self.scaler = scaler
              self.root = None

          def gini_index(self, y):
              _, counts = np.unique(y, return_counts=True)
              probabilities = counts / len(y)
              gini = 1 - np.sum(probabilities**2)
              return gini

          def find_best_split(self, X, y):
              n_features = X.shape[1]
              best_gini = np.inf
              best_feature_idx = None
              best_threshold = None

              for feature_idx in range(n_features):
                  thresholds = np.unique(X[:, feature_idx])
                  for threshold in thresholds:
                      left_indices = X[:, feature_idx] <= threshold
                      right_indices = X[:, feature_idx] > threshold

                      gini_left = self.gini_index(y[left_indices])
                      gini_right = self.gini_index(y[right_indices])
                      gini = (len(y[left_indices]) * gini_left + len(y[right_indices]) * gini_right) / len(y)

                      if gini < best_gini:
                          best_gini = gini
                          best_feature_idx = feature_idx
                          best_threshold = threshold

              return best_feature_idx, best_threshold

          def build_tree(self, X, y, depth=0):
              if depth >= self.max_depth or len(np.unique(y)) == 1 or len(X) < self.max_leaf_nodes:
                  value = np.argmax(np.bincount(y))
                  return Node(value=value)

              feature_idx, threshold = self.find_best_split(X, y)
              left_indices = X[:, feature_idx] <= threshold
              right_indices = X[:, feature_idx] > threshold

              left = self.build_tree(X[left_indices], y[left_indices], depth+1)
              right = self.build_tree(X[right_indices], y[right_indices], depth+1)

              return Node(feature_idx=feature_idx, threshold=threshold, left=left, right=right)

          def fit(self, X, y):
              X_scaled = self.scaler.fit_transform(X)
              self.root = self.build_tree(X_scaled, y)

          def predict(self, X):
              X_scaled = self.scaler.transform(X)
              return np.array([self.traverse_tree(x, self.root) for x in X_scaled])

          def traverse_tree(self, x, node):
              if node.value is not None:
                  return node.value

              if x[node.feature_idx] <= node.threshold:
                  return self.traverse_tree(x, node.left)
              else:
                  return self.traverse_tree(x, node.right)
      # 3. prediksi
      tree_model = DT_Classifier_source_code(max_depth=6, max_leaf_nodes=6)
      tree_model.fit(X_train_scaled, y_train)
      hasil_prediksi = tree_model.predict(fitur)
      hasil_prediksi = int(hasil_prediksi)
      # 4. tampilan outputnya
      txt1="HASIL PREDIKSI PERFORMA PEMAIN ADALAH BURUK 😥"    
      htmlstr1=f"""**<p style='background-color:red;
                              color:white;
                              font-family:Tahoma;
                              font-size:20px;
                              border-radius:3px;
                              line-height:60px;
                              text-align:center;
                              opacity:0.6'>{txt1}</style><br></p>**"""
                                          
      txt2="HASIL PREDIKSI PERFORMA PEMAIN ADALAH NORMAL 👌"    
      htmlstr2=f"""**<p style='background-color:#e0600b;
                              color:white;
                              font-family:Tahoma;
                              font-size:20px;
                              border-radius:3px;
                              line-height:60px;
                              text-align:center;
                              opacity:0.6'>{txt2}</style><br></p>**"""
                                        
      txt3="HASIL PREDIKSI PERFORMA PEMAIN ADALAH BAGUS 👍"    
      htmlstr3=f"""**<p style='background-color:green;
                              color:white;
                              font-size:20px;
                              font-family:Tahoma;
                              border-radius:3px;
                              line-height:60px;
                              text-align:center;
                              opacity:0.6'>{txt3}</style><br></p>**"""
      
      if hasil_prediksi == 0:
        st.markdown(htmlstr1,unsafe_allow_html=True)
        # st.success(f'Hasil prediksi performa pemain adalah Bad 😥')
      elif hasil_prediksi == 1:
        st.markdown(htmlstr2,unsafe_allow_html=True)
        # st.success(f'Hasil prediksi performa pemain adalah **Normal** 👌')
      elif hasil_prediksi == 2:
        st.markdown(htmlstr3,unsafe_allow_html=True)
        # st.success(f'Hasil prediksi performa pemain adalah Good 👍')
      else:
        return 'Nothing'
  
main()