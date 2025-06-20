#claude feyzaimgaa@gmail.com a bak
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import pyodbc
import base64
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMessageBox, QFileDialog,
                             QStatusBar, QDialog, QVBoxLayout, QHBoxLayout,
                             QListWidget, QLabel, QPushButton, QListWidgetItem)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt 
from interface import Ui_MainWindow 
from ultralytics import YOLO
import cvzone
import os


try:
    from deepface import DeepFace
except ImportError:
    print("Uyarı: DeepFace kütüphanesi bulunamadı. Yüz tanıma ve ekleme özellikleri çalışmayabilir.")
    DeepFace = None # DeepFace'i None olarak ayarlıyoruz
import tensorflow as tf
import pandas as pd

# Personel Listesi ve Silme
class PersonelListDialog(QDialog):
    def __init__(self, parent=None, conn=None):
        super().__init__(parent)
        self.setWindowTitle("Personel Yönetimi")
        self.setGeometry(200, 200, 500, 400) 
        self.conn = conn
        self.parent_window = parent # Ana pencereye erişim için

       
        main_layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

     
        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self.show_personel_image) 
        left_layout.addWidget(QLabel("Personeller:"))
        left_layout.addWidget(self.list_widget)


        self.image_label = QLabel("Personel resmi burada görünecek")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(200, 200) 
        self.image_label.setStyleSheet("border: 1px solid gray;")

        self.btn_delete = QPushButton("Seçili Personeli Sil")
        self.btn_delete.clicked.connect(self.delete_selected_personel)

        right_layout.addWidget(self.image_label)
        right_layout.addWidget(self.btn_delete)
        right_layout.addStretch() 

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)

        self.load_personel_list()

    def load_personel_list(self):
        """Veritabanından personel listesini yükler."""
        self.list_widget.clear()
        if not self.conn:
            self.list_widget.addItem("Veritabanı bağlantısı yok.")
            return

        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT PersonelID, Ad, Soyad FROM Personel ORDER BY Ad, Soyad")
            rows = cursor.fetchall()

            if not rows:
                self.list_widget.addItem("Kayıtlı personel bulunamadı.")
                return

            for row in rows:
                personel_id, ad, soyad = row
                item_text = f"{ad} {soyad}"
                list_item = QListWidgetItem(item_text)
                list_item.setData(Qt.UserRole, personel_id) 
                self.list_widget.addItem(list_item)

        except Exception as e:
            self.list_widget.addItem("Personel listesi yüklenemedi.")
            print(f"Personel listesi yükleme hatası: {e}")
            QMessageBox.warning(self, "Hata", f"Personel listesi yüklenirken hata oluştu: {e}")

    def show_personel_image(self, current_item, previous_item):
        """Seçili personelin ilk resmini gösterir."""
        self.image_label.clear()
        self.image_label.setText("Resim yükleniyor...")

        if not current_item:
            self.image_label.setText("Personel seçilmedi")
            return

        personel_id = current_item.data(Qt.UserRole)

        if not self.conn or personel_id is None:
            self.image_label.setText("Resim alınamadı (Veri eksik)")
            return

        try:
            cursor = self.conn.cursor()
            # İlk görüntüyü al
            cursor.execute("""
                SELECT TOP 1 ResimData FROM PersonelGoruntuleri 
                WHERE PersonelID = ? 
                ORDER BY EklemeTarihi ASC
            """, (personel_id,))
            row = cursor.fetchone()

            if row and row[0]:
                img_data = row[0]
                pixmap = QPixmap()
                if pixmap.loadFromData(img_data):
                    scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_label.setPixmap(scaled_pixmap)
                else:
                    self.image_label.setText("Resim formatı geçersiz")
            else:
                self.image_label.setText("Bu personele ait resim bulunamadı")

        except Exception as e:
            self.image_label.setText("Resim yüklenirken hata oluştu")
            print(f"Resim gösterme hatası: {e}")


    def delete_selected_personel(self):
        """Seçili personeli veritabanından siler."""
        current_item = self.list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Uyarı", "Lütfen silmek için bir personel seçin.")
            return

        personel_id = current_item.data(Qt.UserRole)
        personel_adi = current_item.text()

        confirm = QMessageBox.question(self, "Silme Onayı",
                                       f"'{personel_adi}' adlı personeli silmek istediğinizden emin misiniz?\n"
                                       f"Bu işlem personelin tüm giriş/çıkış kayıtlarını da silecektir!",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if confirm == QMessageBox.Yes:
            # Ana penceredeki silme fonksiyonunu çağır
            if self.parent_window and hasattr(self.parent_window, 'delete_personel_from_database'):
                success = self.parent_window.delete_personel_from_database(personel_id)
                if success:
                    QMessageBox.information(self, "Başarılı", f"'{personel_adi}' başarıyla silindi.")
                    # Listeyi yenile
                    self.load_personel_list()
                    self.image_label.clear()
                    self.image_label.setText("Personel resmi burada görünecek")
                else:
                    QMessageBox.critical(self, "Hata", f"'{personel_adi}' silinirken bir hata oluştu.")
            else:
                 QMessageBox.critical(self, "Hata", "Silme işlemi için ana pencere fonksiyonu bulunamadı.")


class KameraPenceresi(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

     # Duygu analizi için yeni özellikler
      #  self.emotion_detection_active = False
        self.last_detected_emotion = None
        self.emotion_history = {}  # Personel ID -> duygu geçmişi
        
        # Duygu analizi butonu ekle (mevcut butonların yanına)
        #self.btn_emotion_analysis = QtWidgets.QPushButton("Duygu Analizi", self)
       # self.btn_emotion_analysis.setGeometry(QtCore.QRect(800, 500, 150, 30))  # Konumu ayarlayın
        #self.btn_emotion_analysis.clicked.connect(self.toggle_emotion_analysis)
        
        # Duygu gösterge labelı ekle
        self.label_emotion = QtWidgets.QLabel("Duygu: -", self)
        self.label_emotion.setGeometry(QtCore.QRect(50, 580, 300, 30))  # Konumu ayarlayın
        self.label_emotion.setStyleSheet("font-size: 14px; font-weight: bold; color: blue;")   


        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # SQL Server bağlantısı
        self.conn = None
        self.setup_database_connection()

        # YOLO yüz tanıma modeli
        try:
            self.facemodel = YOLO('yolov8n-face.pt')
        except Exception as e:
             QMessageBox.critical(self, "Model Yükleme Hatası", f"YOLO modeli (yolov8n-face.pt) yüklenemedi: {e}\nLütfen dosyanın doğru konumda olduğundan emin olun.")
             self.facemodel = None # Modeli None yap

        # DeepFace için model oluştur 
        self.embedding_model = None
        if DeepFace: # Eğer deepFace import edilebildiyse modeli yükler
            try:
                print("DeepFace kullanılabilir.")
            except Exception as e:
                print(f"DeepFace model yapısı oluşturma hatası (göz ardı edilebilir): {e}")
        else:
             print("DeepFace kütüphanesi yüklenemediği için yüz embedding/tanıma devre dışı.")

        self.setGeometry(100, 100, 1000, 700)


        self.known_face_names = [] #Tanınan kişilerin isimlerini tutan liste
        self.known_face_embeddings = {}
        self.load_faces_from_database() # Bağlantı kurulduktan sonra yükle

        # Kamera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Kamera Hatası", "Kamera açılamadı. Bağlı olduğundan ve başka bir uygulama tarafından kullanılmadığından emin olun.")
     
        self.timer = QTimer()
        self.timer.timeout.connect(self.kamera_goster)

        self.btn_start.clicked.connect(self.kamerayi_baslat)
        self.btn_stop.clicked.connect(self.kamerayi_durdur)
        self.btn_kaydet.clicked.connect(self.personel_resmi_ekle)


        self.btn_manage_personnel = QtWidgets.QPushButton("Personel Yönetimi", self)
        self.btn_manage_personnel.setGeometry(QtCore.QRect(800, 460, 150, 30)) 
        self.btn_manage_personnel.clicked.connect(self.open_personnel_management)


        self.btn_recognize = QtWidgets.QPushButton("Tanıma Yap", self)
        self.btn_recognize.setGeometry(QtCore.QRect(800, 540, 150, 30))  
        self.btn_recognize.clicked.connect(self.trigger_recognition)
        
        
        self.figure = Figure(figsize=(6,4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        self.canvas.setGeometry(10, 10, 780, 480)
        self.canvas.hide()
        
# =============================================================================
#         self.personnel_list_widget = QtWidgets.QListWidget(self)
#         self.personnel_list_widget.setGeometry(800, 50, 150, 350)
#         self.personnel_list_widget.itemClicked.connect(self.personel_secildi)
# =============================================================================
        
        # Personel listesi
        self.personnel_list_widget = QtWidgets.QListWidget(self)
        self.personnel_list_widget.setGeometry(800, 50, 150, 350)
        self.personnel_list_widget.itemClicked.connect(self.personel_secildi)
        
        # Kişisel Analiz Butonu
        self.btn_person_analysis = QtWidgets.QPushButton("Kişisel Analiz", self)
        self.btn_person_analysis.setGeometry(QtCore.QRect(800, 420, 150, 30))
        self.btn_person_analysis.clicked.connect(self.trigger_person_analysis)
        
        # Seçilen kişiyi tutmak için alanlar
        self.selected_person_id = None
        self.selected_person_name = None

        self.btn_general_analysis = QtWidgets.QPushButton("Genel Analiz", self)
        self.btn_general_analysis.setGeometry(QtCore.QRect(800, 500, 150, 30))
        self.btn_general_analysis.clicked.connect(self.general_emotion_analysis)
        
        self.btn_back = QtWidgets.QPushButton("Geri", self)
        self.btn_back.setGeometry(QtCore.QRect(800, 580, 150, 30))
        self.btn_back.clicked.connect(self.back_to_main_view)
        self.btn_back.setVisible(False) 
# =============================================================================
#         self.btn_analyze = QtWidgets.QPushButton("Analiz", self)
#         self.btn_analyze.setGeometry(QtCore.QRect(800, 500, 150, 30))
#         self.btn_analyze.clicked.connect(self.show_person_emotion_analysis)
# =============================================================================
         
# =============================================================================
#         self.btn_analyze = QtWidgets.QPushButton("Analiz", self)
#         self.btn_analyze.setGeometry(QtCore.QRect(800, 500, 150, 30))
#         self.btn_analyze.clicked.connect(self.show_person_emotion_analysis)
#     
#         self.btn_general_analysis = QtWidgets.QPushButton("Genel Analiz", self)
#         self.btn_general_analysis.setGeometry(800, 420, 150, 30)
#         self.btn_general_analysis.clicked.connect(self.general_emotion_analysis)
# =============================================================================
        
    
        self.recognition_active = False

        # Tablo başlıklarını ayarlamak için koddan ayarladım
        self.table_log.setColumnCount(4) 
        self.table_log.setHorizontalHeaderLabels(["Personel", "Zaman", "Durum", "Duygu"])
        self.table_log.setColumnWidth(0, 120) 
        self.table_log.setColumnWidth(1, 150)  
        self.table_log.setColumnWidth(2, 80)  
        self.table_log.setColumnWidth(3, 120)  
        self.table_log.horizontalHeader().setStretchLastSection(True)  
    
        self.table_log.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)  
        self.table_log.verticalHeader().setVisible(False)  

        # Daha önce eklenen isimleri kaydetmek için
        self.giris_kaydi = set()
        self.cikis_kaydi = set()

        # VGG-Face için önerilen eşik ~0.40 civarıdır (distance için). Benzerlik (similarity) için ~0.60.
        self.recognition_threshold = 0.4 


        self.last_recognized_face = None
        self.last_recognition_time = None
        

    #Personel Yönetimi Penceresini Açan Fonksiyon
    def open_personnel_management(self):
        """Personel listeleme ve silme dialoğunu açar."""
        if not self.conn:
            QMessageBox.warning(self, "Bağlantı Hatası", "Veritabanı bağlantısı yok!")
            return

        dialog = PersonelListDialog(self, self.conn)
        dialog.exec_()


    def trigger_recognition(self):
        """Tanıma işlemini tetikler - artık duygu analizi de burada yapılacak"""
        if not self.timer.isActive():
            QMessageBox.warning(self, "Uyarı", "Önce kamerayı başlatın!")
            return

        # Eğer DeepFace yoksa tanıma yapılamaz
        if not DeepFace:
            QMessageBox.warning(self, "Uyarı", "DeepFace kütüphanesi yüklenemediği için yüz tanıma ve duygu analizi yapılamıyor.")
            return

        self.recognition_active = True
        self.btn_recognize.setEnabled(False)
        self.btn_recognize.setText("Tanıma ve Duygu Analizi...")
        self.statusBar.showMessage("Yüz tanıma ve duygu analizi aktif...")
        # 5 saniye içinde tanıma yapılmazsa butonu sıfırla
        QTimer.singleShot(5000, self.reset_recognition_flag)

    def reset_recognition_flag(self):
        """Tanıma bayrağını sıfırla"""
        if self.recognition_active:
            self.recognition_active = False
            self.btn_recognize.setEnabled(True)
            self.btn_recognize.setText("Tanıma Yap")
            self.statusBar.showMessage("Hazır", 2000)
                
    def setup_database_connection(self):
        """SQL Server bağlantısını kur"""
        try:
            server = 'localhost' 
            database = 'face_rec_db_2'
    
            connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'
       
            self.conn = pyodbc.connect(connection_string)
            print("Veritabanı bağlantısı kuruldu.")
            self.statusBar.showMessage("Veritabanı bağlantısı başarılı.", 3000)
    
        except pyodbc.Error as ex:
             sqlstate = ex.args[0]
             if sqlstate == '08001':
                 QMessageBox.critical(self, "Veritabanı Hatası", f"SQL Server bağlantısı kurulamadı. Sunucu çalışıyor mu?\n{ex}")
             elif sqlstate == '28000':
                 QMessageBox.critical(self, "Veritabanı Hatası", f"Geçersiz kullanıcı adı veya şifre.\n{ex}")
             elif sqlstate == '42S02':
                 QMessageBox.critical(self, "Veritabanı Hatası", f"Veritabanı bulunamadı: '{database}'.\n{ex}")
             else:
                 QMessageBox.critical(self, "Veritabanı Hatası", f"Veritabanı hatası: {ex}")
             print(f"Veritabanı hatası: {ex}")
             self.conn = None # Bağlantı başarısızsa None yap
        except Exception as e:
            QMessageBox.critical(self, "Veritabanı Hatası", f"Beklenmedik bir veritabanı hatası oluştu: {str(e)}")
            print(f"Beklenmedik veritabanı hatası: {e}")
            self.conn = None
    
    
        def check_database_records(self):
            """Veritabanındaki kayıtları kontrol et (Debug Amaçlı)"""
            if not self.conn:
                QMessageBox.warning(self, "Bağlantı Hatası", "Veritabanı bağlantısı yok!")
                return
    
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM Personel")
                count = cursor.fetchone()[0]
                print(f"Personel tablosunda {count} kayıt var (Debug)")
    
                cursor.execute("SELECT TOP 5 PersonelID, Ad, Soyad FROM Personel ORDER BY PersonelID DESC")
                rows = cursor.fetchall()
    
                if rows:
                    message = f"Personel tablosunda {count} kayıt bulundu.\n\nSon eklenen 5 kayıt:\n"
                    for row in rows:
                        message += f"ID: {row[0]}, Ad: {row[1]}, Soyad: {row[2]}\n"
                else:
                    message = "Personel tablosunda hiç kayıt bulunamadı."
    
                QMessageBox.information(self, "Veritabanı Kayıtları (Debug)", message)
    
            except Exception as e:
                QMessageBox.critical(self, "Sorgu Hatası", f"Veritabanı kontrol hatası (Debug): {str(e)}")
                print(f"Veritabanı kontrol hatası (Debug): {e}")

    #Personel Silme Fonksiyonu
    def delete_personel_from_database(self, personel_id):
        """Verilen ID'ye sahip personeli ve ilgili giriş/çıkış kayıtlarını siler."""
        if not self.conn:
            print("Silme işlemi için veritabanı bağlantısı yok.")
            return False
        if not personel_id:
            print("Silinecek personel ID'si belirtilmedi.")
            return False

        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM Personel WHERE PersonelID = ?", (personel_id,))
            self.conn.commit()
            print(f"Personel ID {personel_id} başarıyla silindi.")
            # Bellekteki yüz verilerini güncellemek için
            self.load_faces_from_database()
            return True
        except Exception as e:
            print(f"Personel silme hatası (ID: {personel_id}): {e}")
            try:
                self.conn.rollback() # Hata durumunda işlemi geri al
                print("İşlem rollback edildi.")
            except Exception as rb_e:
                print(f"Rollback sırasında hata: {rb_e}")
            return False

    def load_faces_from_database(self):
            """Veritabanından personel yüz verilerini yükle"""
            if not self.conn:
                print("Veritabanı bağlantısı yok, yüz verileri yüklenemiyor.")
                return
    
            try:
                cursor = self.conn.cursor()
                # Personel bilgilerini ve tüm görüntülerini al
                cursor.execute("""
                    SELECT p.PersonelID, p.Ad, p.Soyad, pg.YuzEmbedding 
                    FROM Personel p 
                    INNER JOIN PersonelGoruntuleri pg ON p.PersonelID = pg.PersonelID
                    WHERE pg.YuzEmbedding IS NOT NULL
                    ORDER BY p.PersonelID
                """)
                rows = cursor.fetchall()
    
                self.known_face_names = []
                self.known_face_embeddings = {}
    
                # Her personel için birden fazla embedding saklayabilmek için
                personel_embeddings = {}
    
                for row in rows:
                    personel_id, ad, soyad, yuz_embedding_bytes = row
    
                    if yuz_embedding_bytes:
                        try:
                            yuz_embedding = np.frombuffer(yuz_embedding_bytes, dtype=np.float32)
                            personel_adi = f"{ad} {soyad}"
                            
                            # Aynı personelin birden fazla embedding'ini grupla
                            if personel_adi not in personel_embeddings:
                                personel_embeddings[personel_adi] = {
                                    'embeddings': [],
                                    'personel_id': personel_id
                                }
                            
                            personel_embeddings[personel_adi]['embeddings'].append(yuz_embedding)
                      
                        except Exception as deserialize_e:
                            print(f"Personel {ad} {soyad} (ID: {personel_id}) için embedding deserialize edilemedi: {deserialize_e}")
    
                # Her personel için ortalama embedding hesapla veya en iyi eşleşeni kullan
                for personel_adi, data in personel_embeddings.items():
                    embeddings = data['embeddings']
                    if embeddings:
                        # Ortalama embedding hesapla (daha iyi tanıma için)
                        avg_embedding = np.mean(embeddings, axis=0)
                        
                        self.known_face_names.append(personel_adi)
                        self.known_face_embeddings[personel_adi] = {
                            'embedding': avg_embedding,
                            'all_embeddings': embeddings,  # Tüm embedding'leri de sakla
                            'personel_id': data['personel_id']
                        }
    
                print(f"Toplam {len(self.known_face_names)} geçerli personel yüz verisi belleğe yüklendi.")
    
            except Exception as e:
                print(f"Yüz verisi yükleme hatası: {e}")
                QMessageBox.warning(self, "Yükleme Hatası", f"Veritabanından yüz verileri yüklenirken hata oluştu: {e}")
    def personel_resmi_ekle(self):
            """Yeni personel eklemek için birden fazla fotoğrafını seç ve veritabanına kaydet"""
            if not DeepFace:
                 QMessageBox.warning(self, "Eksik Kütüphane", "DeepFace kütüphanesi bulunamadığı için personel eklenemiyor.")
                 return
            if not self.facemodel:
                QMessageBox.warning(self, "Eksik Model", "YOLO yüz tespit modeli yüklenemediği için personel eklenemiyor.")
                return
            if not self.conn:
                 QMessageBox.critical(self, "Hata", "Veritabanı bağlantısı yok!")
                 return
    
            try:
                # Personel bilgilerini al
                ad, ok1 = QtWidgets.QInputDialog.getText(self, "Personel Bilgisi", "Adı:")
                if not ok1 or not ad.strip():
                    return
    
                soyad, ok2 = QtWidgets.QInputDialog.getText(self, "Personel Bilgisi", "Soyadı:")
                if not ok2 or not soyad.strip():
                    return
    
                ad = ad.strip()
                soyad = soyad.strip()
    
                # Çoklu dosya seçimi için dialog
                file_names, _ = QFileDialog.getOpenFileNames(
                    self, 
                    f"{ad} {soyad} için Personel Resimlerini Seçin (3-5 farklı açı önerilir)", 
                    "", 
                    "Resim Dosyaları (*.png *.jpg *.jpeg)"
                )
    
                if not file_names:
                    return
    
                if len(file_names) < 2:
                    reply = QMessageBox.question(self, "Az Görüntü", 
                        "Daha iyi tanıma için en az 2-3 farklı açıdan fotoğraf önerilir.\nDevam etmek istiyor musunuz?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if reply != QMessageBox.Yes:
                        return
    
                # Önce personeli veritabanına ekle
                personel_id = self.save_personel_basic_info(ad, soyad)
                if not personel_id:
                    QMessageBox.critical(self, "Hata", "Personel temel bilgileri kaydedilemedi!")
                    return
    
                valid_images = 0
                failed_images = []
    
                # Her resim için işlem yap
                for i, file_name in enumerate(file_names):
                    try:
                        img = cv2.imread(file_name)
                        if img is None:
                            failed_images.append(f"{os.path.basename(file_name)} - Okunamadı")
                            continue
    
                        # Yüz tespiti
                        results = self.facemodel.predict(img, conf=0.4)[0]
                        boxes = results.boxes
    
                        if len(boxes) == 0:
                            failed_images.append(f"{os.path.basename(file_name)} - Yüz bulunamadı")
                            continue
    
                        # En büyük yüzü al (birden fazla yüz varsa)
                        best_box = max(boxes, key=lambda box: (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]))
                        x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
                        face_img = img[y1:y2, x1:x2]
    
                        # Yüz embedding'i çıkar
                        face_embedding = self.get_face_embedding(face_img)
                        if face_embedding is None:
                            failed_images.append(f"{os.path.basename(file_name)} - Embedding çıkarılamadı")
                            continue
    
                        # Bu görüntüyü veritabanına kaydet
                        if self.save_personel_image_to_database(personel_id, img, face_embedding):
                            valid_images += 1
                            print(f"Görüntü {i+1}/{len(file_names)} başarıyla kaydedildi")
                        else:
                            failed_images.append(f"{os.path.basename(file_name)} - Veritabanına kaydedilemedi")
    
                    except Exception as e:
                        failed_images.append(f"{os.path.basename(file_name)} - Hata: {str(e)}")
                        print(f"Görüntü işleme hatası ({file_name}): {e}")
    
                # Sonuçları rapor et
                if valid_images > 0:
                    self.load_faces_from_database()  # Yeni eklenen yüzleri belleğe yükle
                    
                    message = f"{ad} {soyad} personeli için {valid_images} görüntü başarıyla eklendi!"
                    if failed_images:
                        message += f"\n\nBaşarısız olan görüntüler:\n" + "\n".join(failed_images)
                    
                    QMessageBox.information(self, "Başarılı", message)
                else:
                    # Hiç görüntü eklenemediğinde personeli sil
                    self.delete_personel_from_database(personel_id)
                    QMessageBox.critical(self, "Hata", 
                        f"Hiçbir görüntü eklenemedi! Personel kaydı silindi.\n\nHatalar:\n" + "\n".join(failed_images))
    
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Personel eklenirken beklenmedik bir hata oluştu: {str(e)}")
                print(f"Personel ekleme hatası: {e}")

    def save_personel_to_database(self, ad, soyad, img, embedding):
        """Personeli veritabanına kaydet"""
        if not self.conn:
            print("Veritabanı bağlantısı yok!")
            return False

        try:
            # Resmi binary'e çevir 
            _, img_encoded = cv2.imencode('.jpg', img)
            img_binary = img_encoded.tobytes() # numpy array'i byte dizisine çevir

            # Embedding'i float32 tipine dönüştür ve binary'e çevir
            if embedding.dtype != np.float32:
                 embedding = embedding.astype(np.float32)
            embedding_binary = embedding.tobytes() # numpy array'i byte dizisine çevir

    
            cursor = self.conn.cursor()
            print(f"Veritabanına ekleniyor: Ad='{ad}', Soyad='{soyad}'")

            # VARBINARY(MAX) için doğrudan byte dizisini gönder
            cursor.execute("""
                INSERT INTO Personel (Ad, Soyad, ResimData, YuzEmbedding)
                VALUES (?, ?, ?, ?)
            """, (ad, soyad, img_binary, embedding_binary))


            self.conn.commit()
            print("İşlem commit edildi, personel başarıyla eklendi!")
            return True

        except pyodbc.Error as db_err:
             print(f"Veritabanına kayıt hatası (pyodbc): {db_err}")
             print(f"SQLSTATE: {db_err.args[0]}")
             print(f"Hata Mesajı: {db_err.args[1]}")
             if hasattr(db_err, 'native_error_code'): # MSSQL için özel hata kodu
                 print(f"Native Error Code: {db_err.native_error_code}")
             try:
                 self.conn.rollback()
                 print("İşlem rollback edildi.")
             except Exception as rb_e:
                print(f"Rollback sırasında hata: {rb_e}")
             return False
        except Exception as e:
            print(f"Genel veritabanına kayıt hatası: {e}")
            return False


    def save_personel_basic_info(self, ad, soyad):
            """Personelin temel bilgilerini veritabanına kaydet ve ID'sini döndür"""
            if not self.conn:
                print("Veritabanı bağlantısı yok!")
                return None
    
            try:
                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT INTO Personel (Ad, Soyad)
                    OUTPUT INSERTED.PersonelID
                    VALUES (?, ?)
                """, (ad, soyad))
                
                personel_id = cursor.fetchone()[0]
                self.conn.commit()
                print(f"Personel temel bilgileri eklendi: {ad} {soyad} (ID: {personel_id})")
                return personel_id
    
            except Exception as e:
                print(f"Personel temel bilgi kaydetme hatası: {e}")
                try:
                    self.conn.rollback()
                except:
                    pass
                return None
    
    def save_personel_image_to_database(self, personel_id, img, embedding):
            """Personelin bir görüntüsünü veritabanına kaydet"""
            if not self.conn:
                print("Veritabanı bağlantısı yok!")
                return False
    
            try:
                # Resmi binary'e çevir
                _, img_encoded = cv2.imencode('.jpg', img)
                img_binary = img_encoded.tobytes()
    
                # Embedding'i binary'e çevir
                if embedding.dtype != np.float32:
                     embedding = embedding.astype(np.float32)
                embedding_binary = embedding.tobytes()
    
                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT INTO PersonelGoruntuleri (PersonelID, ResimData, YuzEmbedding)
                    VALUES (?, ?, ?)
                """, (personel_id, img_binary, embedding_binary))
    
                self.conn.commit()
                return True
    
            except Exception as e:
                print(f"Görüntü kaydetme hatası: {e}")
                try:
                    self.conn.rollback()
                except:
                    pass
                return False


    def get_face_embedding(self, face_img):
        """DeepFace ile yüz embedding'i çıkar"""
        if not DeepFace: # DeepFace yoksa None dön
            print("DeepFace kütüphanesi bulunamadığı için embedding çıkarılamıyor.")
            return None

        try:
            # DeepFace'in beklentisi genellikle RGB formatıdır.
            # OpenCV varsayılan olarak BGR okur.
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

          
            embedding_objs = DeepFace.represent(
                img_path=face_rgb,          # RGB numpy array veriyoruz
                model_name="VGG-Face",      # Kullandığımız model
                enforce_detection=False,    # Zaten yüzü kırptık, tekrar tespit etmesin
                detector_backend="skip"     # Tespit adımını atla
            )

            # Sonucun beklenen formatta olup olmadığını kontrol et
            if isinstance(embedding_objs, list) and len(embedding_objs) > 0:
                first_result = embedding_objs[0]
                if isinstance(first_result, dict) and "embedding" in first_result:
                    # Embedding listesini numpy array'e çevirfloat32 olmalı
                    embedding_vector = np.array(first_result["embedding"], dtype=np.float32)
                    return embedding_vector
                else:
                    print("Embedding çıktısı beklenmedik formatta (dict içinde 'embedding' anahtarı yok):", first_result)
                    return None
            else:
                 print("Embedding çıktısı beklenmedik formatta (liste değil veya boş):", embedding_objs)
                 return None

        except ValueError as ve:
             # DeepFace yüz bulamadığında ValueError fırlatabilir 
             print(f"Embedding çıkarma sırasında ValueError: {ve}")
             return None
        except Exception as e:
            print(f"Embedding çıkarma sırasında genel hata: {e}")
            return None


    def recognize_face_with_embedding(self, current_embedding):
            """Verilen embedding'i veritabanındaki kayıtlı embedding'lerle karşılaştırır."""
            if not self.known_face_embeddings:
                return "Unknown", None
            if current_embedding is None:
                 return "Unknown", None
    
            best_match_name = "Unknown"
            best_match_id = None
            min_distance = float('inf')
    
            for name, data in self.known_face_embeddings.items():
                personel_id = data.get('personel_id')
                
                # Önce ortalama embedding ile dene
                avg_embedding = data.get('embedding')
                if avg_embedding is not None and avg_embedding.shape == current_embedding.shape:
                    try:
                        dot_product = np.dot(current_embedding, avg_embedding)
                        norm_current = np.linalg.norm(current_embedding)
                        norm_stored = np.linalg.norm(avg_embedding)
    
                        if norm_current > 0 and norm_stored > 0:
                            similarity = dot_product / (norm_current * norm_stored)
                            distance = 1 - similarity
    
                            if distance < min_distance:
                                min_distance = distance
                                best_match_name = name
                                best_match_id = personel_id
    
                    except Exception as e:
                        print(f"Ortalama embedding karşılaştırma hatası: {e}")
    
                # Eğer ortalama embedding yeterince iyi değilse, tüm embedding'leri dene
                all_embeddings = data.get('all_embeddings', [])
                for stored_embedding in all_embeddings:
                    if stored_embedding.shape != current_embedding.shape:
                        continue
                    try:
                        dot_product = np.dot(current_embedding, stored_embedding)
                        norm_current = np.linalg.norm(current_embedding)
                        norm_stored = np.linalg.norm(stored_embedding)
    
                        if norm_current > 0 and norm_stored > 0:
                            similarity = dot_product / (norm_current * norm_stored)
                            distance = 1 - similarity
    
                            if distance < min_distance:
                                min_distance = distance
                                best_match_name = name
                                best_match_id = personel_id
    
                    except Exception as e:
                        continue
    
            # Eşiği biraz daha esnek yap (çoklu görüntü olduğu için)
            if min_distance < (self.recognition_threshold + 0.1):  # 0.5 yerine 0.6 eşik
                return best_match_name, best_match_id
            else:
                return "Unknown", None


    def kamerayi_baslat(self):
        if not self.cap.isOpened():
             QMessageBox.critical(self, "Kamera Hatası", "Kamera başlatılamıyor. Lütfen bağlantıyı kontrol edin.")
             return
        self.timer.start(30) # ~33 FPS için interval
        self.statusBar.showMessage("Kamera başlatıldı. Tanıma yapmak için 'Tanıma Yap' butonunu kullanın.", 5000)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)


    def kamerayi_durdur(self):
        self.timer.stop()
        self.label_camera.clear()
        self.label_camera.setText("Kamera Kapalı") 
        self.statusBar.showMessage("Kamera durduruldu.", 3000)
        self.reset_recognition_flag()
        # Duygu label'ını da temizle
        self.label_emotion.setText("Duygu: -")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)


    def kaydet_giris_cikis(self, personel_id, durum):
        """Personel giriş-çıkış kaydını veritabanına ekle"""
        if not self.conn:
             print("Giriş/Çıkış kaydı yapılamadı: Veritabanı bağlantısı yok.")
             return False
        if not personel_id:
             print("Giriş/Çıkış kaydı yapılamadı: Geçersiz Personel ID.")
             return False

        try:
            cursor = self.conn.cursor()
            now = datetime.now()
            cursor.execute("""
                INSERT INTO GirisCikis (PersonelID, KayitZamani, Durum)
                VALUES (?, ?, ?)
            """, (personel_id, now, durum))
            self.conn.commit()
            print(f"Kayıt başarılı: Personel ID {personel_id}, Durum: {durum}, Zaman: {now}")
            return True
        except Exception as e:
            print(f"Giriş-çıkış kayıt hatası (Personel ID: {personel_id}): {e}")
            try:
                self.conn.rollback()
            except Exception as rb_e:
                 print(f"Giriş/Çıkış kaydı rollback hatası: {rb_e}")
            return False


    def get_durum(self, personel_id):
        """Personelin son durumunu (Giris/Cikis) kontrol et, ona göre tersini döndür."""
        if not self.conn or not personel_id:
            return "Giris" 

        try:
            cursor = self.conn.cursor()
            # Bu personele ait en son kaydı al
            cursor.execute("""
                SELECT TOP 1 Durum FROM GirisCikis
                WHERE PersonelID = ?
                ORDER BY KayitZamani DESC
            """, (personel_id,))
            row = cursor.fetchone()

            if row:
                son_durum = row[0]
                # Son durum 'Giris' ise şimdi 'çıkış' olmalı, değilse 'giriş' olmalı.
                return "Cikis" if son_durum == "Giris" else "Giris"
            else:
                # Bu personelin hiç kaydı yoksa, ilk kaydı 'giriş' olmalı.
                return "Giris"
        except Exception as e:
            print(f"Durum kontrol hatası (Personel ID: {personel_id}): {e}")
            return "Giris" # Hata durumunda varsayılan


    def kamera_goster(self):
        """Kameradan görüntü alır, yüzleri algılar, tanıma butonu basıldığında tanıma ve duygu analizi yapar."""
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Kameradan frame alınamadı.")
            return

        try:
            target_w, target_h = 700, 500
            frame_resized = cv2.resize(frame, (target_w, target_h))

            # Yüz algılama
            detected_faces = []
            if self.facemodel:
                face_result = self.facemodel.predict(frame_resized, conf=0.40, verbose=False)[0]
                boxes = face_result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    h, w = y2 - y1, x2 - x1
                    detected_faces.append({'box': [x1, y1, w, h], 'coords': (x1, y1, x2, y2)})

            tanima_yapildi_bu_frame = False

            # Algılanan yüzler üzerinde işlem yap
            for face in detected_faces:
                x1, y1, w, h = face['box']
                x1c, y1c, x2c, y2c = face['coords']

                # Yüz kutusunu çiz
                cvzone.cornerRect(frame_resized, [x1, y1, w, h], l=9, rt=3, colorR=(255, 0, 255))

                # Yüz görüntüsünü kırp
                face_img = frame_resized[y1c:y2c, x1c:x2c]
                if face_img.size == 0:
                    continue

                # Tanıma ve duygu analizi (sadece tanıma butonu basıldığında)
                if self.recognition_active and not tanima_yapildi_bu_frame and w > 30 and h > 30:
                    try:
                        # 1. Önce yüz tanıma yap
                        face_embedding = self.get_face_embedding(face_img)
                        if face_embedding is not None:
                            personel_adi, personel_id = self.recognize_face_with_embedding(face_embedding)
                            now = datetime.now()
                            kayit_zamani = now.strftime("%Y-%m-%d %H:%M:%S")

                            # 2. Duygu analizi yap (tanıma ile birlikte)
                            emotion_result = None
                            emotion_text = ""
                            emotion_color = (255, 255, 255)
                            
                            if w > 50 and h > 50:  # Yeterince büyük yüzler için duygu analizi
                                emotion_result = self.analyze_emotion(face_img)
                                if emotion_result:
                                    emotion_tr = self.translate_emotion_to_turkish(emotion_result['emotion'])
                                    confidence = emotion_result['confidence']
                                    emotion_text = f"{emotion_tr} ({confidence:.1f}%)"
                                    emotion_color = self.get_emotion_color(emotion_result['emotion'])
                                    
                                    # Arayüzde göster
                                    self.label_emotion.setText(f"Duygu: {emotion_text}")
                                    self.last_detected_emotion = emotion_result

                            # 3. Tanıma sonucuna göre işlem yap
                            if personel_adi != "Bilinmiyor" and personel_id is not None:
                                durum = self.get_durum(personel_id)
                                
                                # Giriş/çıkış kaydı
                                if self.kaydet_giris_cikis(personel_id, durum):
                                    # Duygu analizi sonucu varsa kaydet
                                    if emotion_result:
                                        self.save_emotion_to_database(personel_id, emotion_result)
                                    
                                    # Tabloya ekle
                                    satir = self.table_log.rowCount()
                                    self.table_log.insertRow(satir)
                                    
                                    emotion_display = "-"
                                    if emotion_result:
                                        emotion_tr = self.translate_emotion_to_turkish(emotion_result['emotion'])
                                        confidence = emotion_result['confidence']
                                        emotion_display = f"{emotion_tr} ({confidence:.0f}%)"
                                    
                                    item_personel = QtWidgets.QTableWidgetItem(personel_adi)
                                    item_zaman = QtWidgets.QTableWidgetItem(kayit_zamani)
                                    item_durum = QtWidgets.QTableWidgetItem(durum)
                                    item_duygu = QtWidgets.QTableWidgetItem(emotion_display)
                                    
                                    self.table_log.setItem(satir, 0, item_personel)
                                    self.table_log.setItem(satir, 1, item_zaman)
                                    self.table_log.setItem(satir, 2, item_durum)
                                    self.table_log.setItem(satir, 3, item_duygu)  # Yeni duygu sütunu
                                    
                                    # En son eklenen kayıt görünsün
                                    self.table_log.scrollToBottom()
                                    
                                    self.label_name.setText(f"Tanınan Kişi: {personel_adi}")
                                    self.label_time.setText(f"Zaman: {kayit_zamani}")
                                    
                                    status_message = f"{personel_adi} için {durum} kaydı yapıldı."
                                    if emotion_text:
                                        status_message += f" Duygu: {emotion_text}"
                                    self.statusBar.showMessage(status_message, 6000)
                                    color = (0, 255, 0)
                                else:
                                    color = (0, 255, 255)
                            else:
                                self.label_name.setText("Tanınan Kişi: Bilinmiyor")
                                self.label_time.setText(f"Zaman: {kayit_zamani}")
                                status_message = "Tanıdık bir yüz bulunamadı."
                                if emotion_text:
                                    status_message += f" Duygu: {emotion_text}"
                                self.statusBar.showMessage(status_message, 4000)
                                personel_adi = "Bilinmiyor"
                                color = (0, 0, 255)

                            # İsim ve duygu yazma
                            display_text = personel_adi
                            if emotion_text:
                                display_text += f" - {emotion_text}"
                            
                            cv2.putText(frame_resized, display_text, (x1, y1 - 35),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            tanima_yapildi_bu_frame = True
                            self.recognition_active = False
                            self.btn_recognize.setEnabled(True)
                            self.btn_recognize.setText("Tanıma Yap")

                    except Exception as rec_err:
                        print(f"Tanıma hatası: {rec_err}")
                        self.recognition_active = False
                        self.btn_recognize.setEnabled(True)
                        self.btn_recognize.setText("Tanıma Yap")

                # Tanıma aktif değilken sadece "Yüz Algılandı" göster
                elif not self.recognition_active:
                    cv2.putText(frame_resized, "Yuz Algilandi", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 160, 0), 2)

            # Frame'i göster
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.label_camera.setPixmap(QPixmap.fromImage(q_img))

        except Exception as e:
            print(f"kamera_goster fonksiyonunda genel hata: {e}")

    def closeEvent(self, event):
        """Pencere kapatıldığında kaynakları serbest bırak"""
        print("Uygulama kapatılıyor...")
        self.timer.stop()
        self.emotion_detection_active = False
        if self.cap.isOpened():
            self.cap.release() 
            print("Kamera serbest bırakıldı.")
        if self.conn:
            self.conn.close()
            print("Veritabanı bağlantısı kapatıldı.")
        event.accept()

    # def toggle_emotion_analysis(self):
    #     """Duygu analizini aç/kapat"""
    #     if not DeepFace:
    #         QMessageBox.warning(self, "Uyarı", "DeepFace kütüphanesi yüklenemediği için duygu analizi yapılamıyor.")
    #         return
        
    #     if not self.timer.isActive():
    #         QMessageBox.warning(self, "Uyarı", "Önce kamerayı başlatın!")
    #         return
        
    #     self.emotion_detection_active = not self.emotion_detection_active
        
    #     if self.emotion_detection_active:
    #         self.btn_emotion_analysis.setText("Duygu Analizini Durdur")
    #         self.btn_emotion_analysis.setStyleSheet("background-color: red; color: white;")
    #         self.statusBar.showMessage("Duygu analizi aktif...", 3000)
    #     else:
    #         self.btn_emotion_analysis.setText("Duygu Analizi")
    #         self.btn_emotion_analysis.setStyleSheet("")
    #         self.label_emotion.setText("Duygu: -")
    #         self.statusBar.showMessage("Duygu analizi durduruldu.", 3000)
    
    # 3. Duygu analizi fonksiyonu ekleyin:
    
    def analyze_emotion(self, face_img):
        """DeepFace ile duygu analizi yapar"""
        if not DeepFace:
            return None
        
        try:
            # Duygu analizi için resmi RGB formatına çevir
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # DeepFace ile duygu analizi
            result = DeepFace.analyze(
                img_path=face_rgb,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend="skip"
            )
            
            #if isinstance(result, list) and len(result) > 0:
             #   emotions = result[0].get('emotion', {})
             # DeepFace bazen liste bazen sözlük döndürebiliyor
            if isinstance(result, list):
                result = result[0] if result else None
            
            if isinstance(result, dict):
                emotions = result.get('emotion', {})
                if emotions:
                    # En yüksek skorlu duyguyu bul
                    dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                    return {
                        'emotion': dominant_emotion[0],
                        'confidence': dominant_emotion[1],
                        'all_emotions': emotions
                    }
        except Exception as e:
            print(f"Duygu analizi hatası: {e}")
            return None
        
        return None
    
    # 4. Duygu geçmişini kaydetme fonksiyonu:
    
    def save_emotion_to_database(self, personel_id, emotion_data):
        """Duygu analizini veritabanına kaydet"""
        if not self.conn or not personel_id or not emotion_data:
            return False
        
        try:
            cursor = self.conn.cursor()
            now = datetime.now()
            
            # Duygu analizi tablosu yoksa oluşturmak için SQL:
            # CREATE TABLE DuyguAnalizi (
            #     ID INT IDENTITY(1,1) PRIMARY KEY,
            #     PersonelID INT FOREIGN KEY REFERENCES Personel(PersonelID),
            #     Duygu NVARCHAR(50),
            #     GuvenOrani FLOAT,
            #     KayitZamani DATETIME
            # )
            
            cursor.execute("""
                INSERT INTO DuyguAnalizi (PersonelID, Duygu, GuvenOrani, KayitZamani)
                VALUES (?, ?, ?, ?)
            """, (personel_id, emotion_data['emotion'], emotion_data['confidence'], now))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Duygu veritabanı kayıt hatası: {e}")
            return False
    
    # 5. Türkçe duygu çevirisi fonksiyonu:
    
    def translate_emotion_to_turkish(self, emotion):
        """İngilizce duygu isimlerini Türkçeye çevirir"""
        emotion_dict = {
            'angry': 'Kızgın',
            'disgust': 'İğrenme',
            'fear': 'Korku',
            'happy': 'Mutlu',
            'sad': 'Üzgün',
            'surprise': 'Şaşkın',
            'neutral': 'Nötr'
        }
        return emotion_dict.get(emotion.lower(), emotion)
    
    # 6. Duygu rengini belirleme fonksiyonu:
    
    def get_emotion_color(self, emotion):
        """Duyguya göre renk döndürür (BGR formatında)"""
        color_dict = {
            'angry': (0, 0, 255),      # Kırmızı
            'disgust': (0, 128, 0),    # Koyu yeşil
            'fear': (128, 0, 128),     # Mor
            'happy': (0, 255, 0),      # Yeşil
            'sad': (255, 0, 0),        # Mavi
            'surprise': (0, 255, 255), # Sarı
            'neutral': (128, 128, 128) # Gri
        }
        return color_dict.get(emotion.lower(), (255, 255, 255))  # Varsayılan beyaz
    

    def back_to_main_view(self):
        self.canvas.hide()
        self.figure.clear()
        self.canvas.draw()
    
        self.personnel_list_widget.setVisible(False)  # Liste gizlensin
        self.btn_person_analysis.setVisible(True)     # Kişisel analiz butonu görünsün
        self.btn_general_analysis.setVisible(True)    # Genel analiz butonu görünsün
        self.btn_back.setVisible(False)     


    def load_personnel_list(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT PersonelID, Ad, Soyad FROM Personel ORDER BY Ad, Soyad")
            rows = cursor.fetchall()
    
            self.personnel_list_widget.clear()
            self.personel_dict = {}
    
            for personel_id, ad, soyad in rows:
                ad_soyad = f"{ad} {soyad}"
                self.personel_dict[ad_soyad] = personel_id
                print(f"Ekle: {ad_soyad}")
                self.personnel_list_widget.addItem(ad_soyad)
    
        except Exception as e:
            print(f"Personel listesi çekilemedi: {e}")

# =============================================================================
#     def load_personnel_list(self):
#         try:
#             cursor = self.conn.cursor()
#             cursor.execute("SELECT PersonelID, Ad, Soyad FROM Personel ORDER BY Ad, Soyad")  # Personel tablosunda id ve ad soyad varsa
#             rows = cursor.fetchall()
#     
#             self.personnel_list_widget.clear()  # QListWidget örneği
#             self.personel_dict = {}  # ID ile isim eşlemesi için
#     
#             for row in rows:
#                 personel_id, ad_soyad = row
#                 self.personel_dict[ad_soyad] = personel_id
#                 print(f"Ekle: {ad_soyad}")  # Burada eklenen isimleri yazdır
#                 self.personnel_list_widget.addItem(ad_soyad)
# =============================================================================
                # =============================================================================
#                 personel_id, ad_soyad = row
#                 self.personel_dict[ad_soyad] = personel_id
#                 self.personnel_list_widget.addItem(ad_soyad)
#                 print(f"Personel sayısı: {len(rows)}")
# =============================================================================

    
    def personel_secildi(self, item):
        personel_adi = item.text()
        personel_id = self.personel_dict.get(personel_adi)
    
        if not personel_id:
            QtWidgets.QMessageBox.warning(self, "Hata", "Personel bulunamadı.")
            return
    
        self.selected_person_id = personel_id
        self.selected_person_name = personel_adi
        self.show_person_emotion_analysis(personel_id, personel_adi)
        
        self.personnel_list_widget.setVisible(False)

    def trigger_person_analysis(self):
       # if self.selected_person_id and self.selected_person_name:
              self.load_personnel_list()
              self.personnel_list_widget.setVisible(True)
              self.btn_back.setVisible(True)
              self.btn_general_analysis.setVisible(False)
              self.btn_person_analysis.setVisible(False)
# =============================================================================
#             self.show_person_emotion_analysis(self.selected_person_id, self.selected_person_name)
#             self.personnel_list_widget.setVisible(True)  # Listeyi görünür yap
#             self.btn_back.setVisible(True)  # Geri butonu göster
#             self.btn_personal_analysis.setVisible(False)  # Kişisel analiz butonunu gizle
#             self.btn_analyze.setVisible(False) 
# =============================================================================
    #    else:
     #       QtWidgets.QMessageBox.warning(self, "Uyarı", "Lütfen önce listeden bir personel seçiniz.")
        

    def show_person_emotion_analysis(self, personel_id, personel_adi):
        try:
            cursor = self.conn.cursor()
            yedi_gun_once = datetime.now() - timedelta(days=7)
    
            cursor.execute("""
                SELECT Duygu, COUNT(*) as Sayac
                FROM DuyguAnalizi
                WHERE KayitZamani >= ? AND PersonelID = ?
                GROUP BY Duygu
            """, (yedi_gun_once, personel_id))  # Burada kişiye filtreleme geldi
    
            rows = cursor.fetchall()
    
            if not rows:
                QtWidgets.QMessageBox.information(self, "Bilgi", f"{personel_adi} için son 7 güne ait veri bulunamadı.")
                return
    
            duygular = []
            sayaclar = []
            for row in rows:
                duygular.append(self.translate_emotion_to_turkish(row[0]))
                sayaclar.append(row[1])
    
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.bar(duygular, sayaclar, color='lightgreen')
            ax.set_title(f"{personel_adi} - Son 7 Günlük Duygu Dağılımı")
            ax.set_xlabel("Duygular")
            ax.set_ylabel("Adet")
            self.figure.tight_layout()
            self.canvas.draw()
            self.canvas.show()
            
    
        except Exception as e:
            print(f"Kişisel analiz hatası: {e}")
            QtWidgets.QMessageBox.critical(self, "Hata", "Kişiye özel duygu analizi gösterilemedi.")

    def general_emotion_analysis(self):
        try:
            cursor = self.conn.cursor()
            yedi_gun_once = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    
            cursor.execute("""
                SELECT Duygu, COUNT(*) as Sayac
                FROM DuyguAnalizi
                WHERE KayitZamani >= ?
                GROUP BY Duygu
            """, (yedi_gun_once,))
    
            rows = cursor.fetchall()
    
            if not rows:
                QtWidgets.QMessageBox.information(self, "Bilgi", "Son 7 güne ait duygu verisi bulunamadı.")
                return
    
            duygular = []
            sayaclar = []
            for row in rows:
                duygular.append(self.translate_emotion_to_turkish(row[0]))
                sayaclar.append(row[1])
    
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.bar(duygular, sayaclar, color='orange')
            ax.set_title("Genel - Son 7 Günlük Duygu Dağılımı")
            ax.set_xlabel("Duygular")
            ax.set_ylabel("Adet")
            self.figure.tight_layout()
            self.canvas.draw()
            self.canvas.show()
    
            # Arayüz durumunu güncelle
            self.btn_back.setVisible(True)
            self.btn_general_analysis.setVisible(False)
            self.btn_person_analysis.setVisible(False)
            # Personel listesi açıksa gizle
            self.personnel_list_widget.setVisible(False)
    
        except Exception as e:
            print(f"Genel analiz hatası: {e}")
            QtWidgets.QMessageBox.critical(self, "Hata", "Genel duygu analizi gösterilemedi.")

# =============================================================================
#     def show_weekly_emotion_analysis(self):
#         try:
#             cursor = self.conn.cursor()
#             yedi_gun_once = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
#             cursor.execute("""
#                 SELECT Duygu, COUNT(*) as Sayac
#                 FROM DuyguAnalizi
#                 WHERE KayitZamani >= ?
#                 GROUP BY Duygu
#             """, (yedi_gun_once,))
#             rows = cursor.fetchall()
# 
#             if not rows:
#                 QtWidgets.QMessageBox.information(self, "Bilgi", "Son 7 güne ait duygu verisi bulunamadı.")
#                 return
# 
#             duygular = []
#             sayaclar = []
#             for row in rows:
#                 duygular.append(self.translate_emotion_to_turkish(row[0]))
#                 sayaclar.append(row[1])
# 
#             # Figure'u temizle
#             self.figure.clear()
# 
#             # Yeni eksen oluştur
#             ax = self.figure.add_subplot(111)
#             ax.bar(duygular, sayaclar, color='skyblue')
#             ax.set_title("Son 7 Günlük Duygu Dağılımı")
#             ax.set_xlabel("Duygular")
#             ax.set_ylabel("Adet")
#             self.figure.tight_layout()
# 
#             # Canvas'ı güncelle
#             self.canvas.draw()
#             self.canvas.show()
# 
#         except Exception as e:
#             print(f"Haftalık analiz hatası: {e}")
#             QtWidgets.QMessageBox.critical(self, "Hata", "Duygu analizi gösterilemedi.")
# =============================================================================
if __name__ == "__main__":


    app = QApplication(sys.argv)
    pencere = KameraPenceresi()
    pencere.show()
    sys.exit(app.exec_())
    

