# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'design.ui'
##
## Created by: Qt User Interface Compiler version 6.10.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QSpinBox, QLabel, QMainWindow,
    QMenuBar, QPushButton, QSizePolicy, QStatusBar,
    QWidget)

class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        if not mainWindow.objectName():
            mainWindow.setObjectName(u"mainWindow")
        mainWindow.resize(1211, 753)
        self.centralwidget = QWidget(mainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.load_template_button = QPushButton(self.centralwidget)
        self.load_template_button.setObjectName(u"load_template_button")
        self.load_template_button.setGeometry(QRect(20, 20, 251, 26))
        self.w_label = QLabel(self.centralwidget)
        self.w_label.setObjectName(u"w_label")
        self.w_label.setGeometry(QRect(30, 100, 181, 18))
        self.w_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.params_label = QLabel(self.centralwidget)
        self.params_label.setObjectName(u"params_label")
        self.params_label.setGeometry(QRect(90, 60, 111, 20))
        self.params_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thickness_label = QLabel(self.centralwidget)
        self.thickness_label.setObjectName(u"thickness_label")
        self.thickness_label.setGeometry(QRect(30, 150, 181, 18))
        self.thickness_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.aL_label = QLabel(self.centralwidget)
        self.aL_label.setObjectName(u"aL_label")
        self.aL_label.setGeometry(QRect(100, 200, 181, 18))
               
        self.width_doubleSpinBox = QDoubleSpinBox(self.centralwidget)
        self.width_doubleSpinBox.setObjectName(u"width_doubleSpinBox")
        self.width_doubleSpinBox.setMaximum(5000)
        self.width_doubleSpinBox.setGeometry(QRect(220, 90, 71, 27))
        self.thickness_doubleSpinBox = QDoubleSpinBox(self.centralwidget)
        self.thickness_doubleSpinBox.setObjectName(u"thickness_doubleSpinBox")
        self.thickness_doubleSpinBox.setGeometry(QRect(220, 140, 71, 27))
        self.aL_doubleSpinBox = QDoubleSpinBox(self.centralwidget)
        self.aL_doubleSpinBox.setObjectName(u"aL_doubleSpinBox")
        self.aL_doubleSpinBox.setGeometry(QRect(220, 200, 71, 27))
        
        mainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(mainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1211, 23))
        mainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(mainWindow)
        self.statusbar.setObjectName(u"statusbar")
        mainWindow.setStatusBar(self.statusbar)
        
        # Параметры ГРП
        self.skin_label = QLabel(self.centralwidget)
        self.skin_label.setObjectName(u"skin_label")
        self.skin_label.setGeometry(QRect(30, 250, 181, 18))
        self.skin_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.n_label = QLabel(self.centralwidget)
        self.n_label.setObjectName(u"n_label")
        self.n_label.setGeometry(QRect(30, 300, 181, 18))
        self.n_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.skin_doubleSpinBox = QDoubleSpinBox(self.centralwidget)
        self.skin_doubleSpinBox.setObjectName(u"skin_doubleSpinBox")
        self.skin_doubleSpinBox.setGeometry(QRect(220, 240, 71, 27))

        self.n_spinBox = QSpinBox(self.centralwidget)  # Число трещин - целое
        self.n_spinBox.setObjectName(u"n_spinBox")
        self.n_spinBox.setGeometry(QRect(220, 290, 71, 27))
        
        self.retranslateUi(mainWindow)

        QMetaObject.connectSlotsByName(mainWindow)
    # setupUi

    def retranslateUi(self, mainWindow):
        mainWindow.setWindowTitle(QCoreApplication.translate("mainWindow", "MainWindow", None))
        self.params_label.setText(QCoreApplication.translate("mainWindow", "Параметры", None))
        self.load_template_button.setText(QCoreApplication.translate("mainWindow", "Загрузить csv/pq шаблон", None))
        self.w_label.setText(QCoreApplication.translate("mainWindow", "Длина трещины W, м", None))
        self.thickness_label.setText(QCoreApplication.translate("mainWindow", "Толщина пласта h, м", None))
        self.skin_label.setText(QCoreApplication.translate("mainWindow", "Skin", None))
        self.n_label.setText(QCoreApplication.translate("mainWindow", "N (трещины)", None))
        self.aL_label.setText(QCoreApplication.translate("mainWindow", "a/L", None))
    # retranslateUi

