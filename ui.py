# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_ui.ui'
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
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QSpinBox, QStatusBar, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1400, 900)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_3 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.handle_menu = QVBoxLayout()
        self.handle_menu.setObjectName(u"handle_menu")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.load_template_button = QPushButton(self.centralwidget)
        self.load_template_button.setObjectName(u"load_template_button")

        self.horizontalLayout.addWidget(self.load_template_button)

        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)


        self.handle_menu.addLayout(self.horizontalLayout)

        self.params_group = QGroupBox(self.centralwidget)
        self.params_group.setObjectName(u"params_group")
        self.gridLayout = QGridLayout(self.params_group)
        self.gridLayout.setObjectName(u"gridLayout")
        self.w_label = QLabel(self.params_group)
        self.w_label.setObjectName(u"w_label")

        self.gridLayout.addWidget(self.w_label, 0, 0, 1, 1)

        self.width_doubleSpinBox = QDoubleSpinBox(self.params_group)
        self.width_doubleSpinBox.setObjectName(u"width_doubleSpinBox")

        self.gridLayout.addWidget(self.width_doubleSpinBox, 0, 1, 1, 1)

        self.thickness_label = QLabel(self.params_group)
        self.thickness_label.setObjectName(u"thickness_label")

        self.gridLayout.addWidget(self.thickness_label, 1, 0, 1, 1)

        self.thickness_doubleSpinBox = QDoubleSpinBox(self.params_group)
        self.thickness_doubleSpinBox.setObjectName(u"thickness_doubleSpinBox")

        self.gridLayout.addWidget(self.thickness_doubleSpinBox, 1, 1, 1, 1)

        self.aL_label = QLabel(self.params_group)
        self.aL_label.setObjectName(u"aL_label")

        self.gridLayout.addWidget(self.aL_label, 2, 0, 1, 1)

        self.aL_doubleSpinBox = QDoubleSpinBox(self.params_group)
        self.aL_doubleSpinBox.setObjectName(u"aL_doubleSpinBox")

        self.gridLayout.addWidget(self.aL_doubleSpinBox, 2, 1, 1, 1)

        self.skin_label = QLabel(self.params_group)
        self.skin_label.setObjectName(u"skin_label")

        self.gridLayout.addWidget(self.skin_label, 3, 0, 1, 1)

        self.skin_doubleSpinBox = QDoubleSpinBox(self.params_group)
        self.skin_doubleSpinBox.setObjectName(u"skin_doubleSpinBox")

        self.gridLayout.addWidget(self.skin_doubleSpinBox, 3, 1, 1, 1)

        self.n_label = QLabel(self.params_group)
        self.n_label.setObjectName(u"n_label")

        self.gridLayout.addWidget(self.n_label, 4, 0, 1, 1)

        self.n_spinBox = QSpinBox(self.params_group)
        self.n_spinBox.setObjectName(u"n_spinBox")

        self.gridLayout.addWidget(self.n_spinBox, 4, 1, 1, 1)


        self.handle_menu.addWidget(self.params_group)

        self.approx_model_group = QGroupBox(self.centralwidget)
        self.approx_model_group.setObjectName(u"approx_model_group")
        self.gridLayout1 = QGridLayout(self.approx_model_group)
        self.gridLayout1.setObjectName(u"gridLayout1")
        self.approx_train_model_btn = QPushButton(self.approx_model_group)
        self.approx_train_model_btn.setObjectName(u"approx_train_model_btn")

        self.gridLayout1.addWidget(self.approx_train_model_btn, 0, 0, 1, 1)

        self.approx_save_model_btn = QPushButton(self.approx_model_group)
        self.approx_save_model_btn.setObjectName(u"approx_save_model_btn")

        self.gridLayout1.addWidget(self.approx_save_model_btn, 0, 1, 1, 1)

        self.approx_load_model_btn = QPushButton(self.approx_model_group)
        self.approx_load_model_btn.setObjectName(u"approx_load_model_btn")

        self.gridLayout1.addWidget(self.approx_load_model_btn, 1, 0, 1, 1)

        self.approx_model_status_label = QLabel(self.approx_model_group)
        self.approx_model_status_label.setObjectName(u"approx_model_status_label")
        self.approx_model_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout1.addWidget(self.approx_model_status_label, 1, 1, 1, 1)


        self.handle_menu.addWidget(self.approx_model_group)

        self.interpolar_model_group = QGroupBox(self.centralwidget)
        self.interpolar_model_group.setObjectName(u"interpolar_model_group")
        self.gridLayout2 = QGridLayout(self.interpolar_model_group)
        self.gridLayout2.setObjectName(u"gridLayout2")
        self.interpolar_train_model_btn = QPushButton(self.interpolar_model_group)
        self.interpolar_train_model_btn.setObjectName(u"interpolar_train_model_btn")

        self.gridLayout2.addWidget(self.interpolar_train_model_btn, 0, 0, 1, 1)

        self.interpolar_load_model_btn = QPushButton(self.interpolar_model_group)
        self.interpolar_load_model_btn.setObjectName(u"interpolar_load_model_btn")

        self.gridLayout2.addWidget(self.interpolar_load_model_btn, 1, 0, 1, 1)

        self.interpolar_model_status_label = QLabel(self.interpolar_model_group)
        self.interpolar_model_status_label.setObjectName(u"interpolar_model_status_label")
        self.interpolar_model_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout2.addWidget(self.interpolar_model_status_label, 1, 1, 1, 1)

        self.interpolar_save_model_btn = QPushButton(self.interpolar_model_group)
        self.interpolar_save_model_btn.setObjectName(u"interpolar_save_model_btn")

        self.gridLayout2.addWidget(self.interpolar_save_model_btn, 0, 1, 1, 1)


        self.handle_menu.addWidget(self.interpolar_model_group)

        self.report_group = QGroupBox(self.centralwidget)
        self.report_group.setObjectName(u"report_group")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.report_group.sizePolicy().hasHeightForWidth())
        self.report_group.setSizePolicy(sizePolicy)
        self.report_group.setMinimumSize(QSize(0, 149))
        self.vboxLayout = QVBoxLayout(self.report_group)
        self.vboxLayout.setObjectName(u"vboxLayout")
        self.text_report = QTextEdit(self.report_group)
        self.text_report.setObjectName(u"text_report")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.text_report.sizePolicy().hasHeightForWidth())
        self.text_report.setSizePolicy(sizePolicy1)
        self.text_report.setMinimumSize(QSize(0, 176))
        self.text_report.setReadOnly(True)

        self.vboxLayout.addWidget(self.text_report)


        self.handle_menu.addWidget(self.report_group)


        self.horizontalLayout_2.addLayout(self.handle_menu)

        self.tab_widget = QTabWidget(self.centralwidget)
        self.tab_widget.setObjectName(u"tab_widget")
        sizePolicy1.setHeightForWidth(self.tab_widget.sizePolicy().hasHeightForWidth())
        self.tab_widget.setSizePolicy(sizePolicy1)
        self.timeseries_tab = QWidget()
        self.timeseries_tab.setObjectName(u"timeseries_tab")
        self.hboxLayout = QHBoxLayout(self.timeseries_tab)
        self.hboxLayout.setObjectName(u"hboxLayout")
        self.timeseries_controls_placeholder = QWidget(self.timeseries_tab)
        self.timeseries_controls_placeholder.setObjectName(u"timeseries_controls_placeholder")

        self.hboxLayout.addWidget(self.timeseries_controls_placeholder)

        self.dimensionless_plot_placeholder = QWidget(self.timeseries_tab)
        self.dimensionless_plot_placeholder.setObjectName(u"dimensionless_plot_placeholder")

        self.hboxLayout.addWidget(self.dimensionless_plot_placeholder)

        self.tab_widget.addTab(self.timeseries_tab, "")
        self.grp_tab = QWidget()
        self.grp_tab.setObjectName(u"grp_tab")
        self.vboxLayout1 = QVBoxLayout(self.grp_tab)
        self.vboxLayout1.setObjectName(u"vboxLayout1")
        self.tab_widget.addTab(self.grp_tab, "")
        self.type_curves_tab = QWidget()
        self.type_curves_tab.setObjectName(u"type_curves_tab")
        self.vboxLayout2 = QVBoxLayout(self.type_curves_tab)
        self.vboxLayout2.setObjectName(u"vboxLayout2")
        self.type_curves_plot_placeholder = QWidget(self.type_curves_tab)
        self.type_curves_plot_placeholder.setObjectName(u"type_curves_plot_placeholder")

        self.vboxLayout2.addWidget(self.type_curves_plot_placeholder)

        self.tab_widget.addTab(self.type_curves_tab, "")
        self.results_tab = QWidget()
        self.results_tab.setObjectName(u"results_tab")
        self.vboxLayout3 = QVBoxLayout(self.results_tab)
        self.vboxLayout3.setObjectName(u"vboxLayout3")
        self.results_text = QTextEdit(self.results_tab)
        self.results_text.setObjectName(u"results_text")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.results_text.sizePolicy().hasHeightForWidth())
        self.results_text.setSizePolicy(sizePolicy2)

        self.vboxLayout3.addWidget(self.results_text)

        self.export_report_btn = QPushButton(self.results_tab)
        self.export_report_btn.setObjectName(u"export_report_btn")

        self.vboxLayout3.addWidget(self.export_report_btn)

        self.tab_widget.addTab(self.results_tab, "")

        self.horizontalLayout_2.addWidget(self.tab_widget)


        self.horizontalLayout_3.addLayout(self.horizontalLayout_2)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.tab_widget.setCurrentIndex(3)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.load_template_button.setText(QCoreApplication.translate("MainWindow", u"\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u044c csv/pq \u0448\u0430\u0431\u043b\u043e\u043d", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.params_group.setTitle(QCoreApplication.translate("MainWindow", u"\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b", None))
        self.w_label.setText(QCoreApplication.translate("MainWindow", u"\u0414\u043b\u0438\u043d\u0430 \u0442\u0440\u0435\u0449\u0438\u043d\u044b W, \u043c", None))
        self.thickness_label.setText(QCoreApplication.translate("MainWindow", u"\u0422\u043e\u043b\u0449\u0438\u043d\u0430 \u043f\u043b\u0430\u0441\u0442\u0430 h, \u043c", None))
        self.aL_label.setText(QCoreApplication.translate("MainWindow", u"a/L", None))
        self.skin_label.setText(QCoreApplication.translate("MainWindow", u"Skin", None))
        self.n_label.setText(QCoreApplication.translate("MainWindow", u"N (\u0442\u0440\u0435\u0449\u0438\u043d\u044b)", None))
        self.approx_model_group.setTitle(QCoreApplication.translate("MainWindow", u"\u041c\u043e\u0434\u0435\u043b\u044c \u0430\u043f\u043f\u0440\u043e\u043a\u0441\u0438\u043c\u0430\u0446\u0438\u0438", None))
        self.approx_train_model_btn.setText(QCoreApplication.translate("MainWindow", u"\u041e\u0431\u0443\u0447\u0438\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c", None))
        self.approx_save_model_btn.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c", None))
        self.approx_load_model_btn.setText(QCoreApplication.translate("MainWindow", u"\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c", None))
        self.approx_model_status_label.setText(QCoreApplication.translate("MainWindow", u"\u041c\u043e\u0434\u0435\u043b\u044c \u043d\u0435 \u043e\u0431\u0443\u0447\u0435\u043d\u0430", None))
        self.interpolar_model_group.setTitle(QCoreApplication.translate("MainWindow", u"\u041c\u043e\u0434\u0435\u043b\u044c \u0438\u043d\u0442\u0435\u0440\u043f\u043e\u043b\u044f\u0446\u0438\u0438", None))
        self.interpolar_train_model_btn.setText(QCoreApplication.translate("MainWindow", u"\u041e\u0431\u0443\u0447\u0438\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c", None))
        self.interpolar_load_model_btn.setText(QCoreApplication.translate("MainWindow", u"\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c", None))
        self.interpolar_model_status_label.setText(QCoreApplication.translate("MainWindow", u"\u041c\u043e\u0434\u0435\u043b\u044c \u043d\u0435 \u043e\u0431\u0443\u0447\u0435\u043d\u0430", None))
        self.interpolar_save_model_btn.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c", None))
        self.report_group.setTitle(QCoreApplication.translate("MainWindow", u"\u041e\u0442\u0447\u0451\u0442", None))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.timeseries_tab), QCoreApplication.translate("MainWindow", u"\u0411\u0435\u0437\u0440\u0430\u0437\u043c\u0435\u0440\u043d\u044b\u0435 \u043a\u0440\u0438\u0432\u044b\u0435", None))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.grp_tab), QCoreApplication.translate("MainWindow", u"\u0410\u043d\u0430\u043b\u0438\u0437 \u0413\u0420\u041f", None))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.type_curves_tab), QCoreApplication.translate("MainWindow", u"\u042d\u0442\u0430\u043b\u043e\u043d\u043d\u044b\u0435 \u043a\u0440\u0438\u0432\u044b\u0435", None))
        self.export_report_btn.setText(QCoreApplication.translate("MainWindow", u"\u042d\u043a\u0441\u043f\u043e\u0440\u0442 \u043e\u0442\u0447\u0435\u0442\u0430", None))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.results_tab), QCoreApplication.translate("MainWindow", u"\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b", None))
    # retranslateUi

