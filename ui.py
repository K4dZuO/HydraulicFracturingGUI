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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLayout, QMainWindow, QPushButton,
    QSizePolicy, QSpinBox, QStatusBar, QTabWidget,
    QTableView, QTextEdit, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.setWindowModality(Qt.WindowModality.NonModal)
        MainWindow.resize(1920, 1080)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_3 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.left_panel = QGroupBox(self.centralwidget)
        self.left_panel.setObjectName(u"left_panel")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.left_panel.sizePolicy().hasHeightForWidth())
        self.left_panel.setSizePolicy(sizePolicy)
        self.verticalLayout = QVBoxLayout(self.left_panel)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.load_template_button = QPushButton(self.left_panel)
        self.load_template_button.setObjectName(u"load_template_button")

        self.horizontalLayout.addWidget(self.load_template_button)

        self.load_file_label = QLabel(self.left_panel)
        self.load_file_label.setObjectName(u"load_file_label")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.load_file_label.sizePolicy().hasHeightForWidth())
        self.load_file_label.setSizePolicy(sizePolicy1)
        self.load_file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout.addWidget(self.load_file_label)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.params_group = QGroupBox(self.left_panel)
        self.params_group.setObjectName(u"params_group")
        self.gridLayout = QGridLayout(self.params_group)
        self.gridLayout.setObjectName(u"gridLayout")
        self.thickness_label = QLabel(self.params_group)
        self.thickness_label.setObjectName(u"thickness_label")

        self.gridLayout.addWidget(self.thickness_label, 1, 0, 1, 1)

        self.n_spinBox = QSpinBox(self.params_group)
        self.n_spinBox.setObjectName(u"n_spinBox")
        self.n_spinBox.setReadOnly(True)
        self.n_spinBox.setMaximum(10000)

        self.gridLayout.addWidget(self.n_spinBox, 4, 1, 1, 1)

        self.skin_label = QLabel(self.params_group)
        self.skin_label.setObjectName(u"skin_label")

        self.gridLayout.addWidget(self.skin_label, 3, 0, 1, 1)

        self.n_label = QLabel(self.params_group)
        self.n_label.setObjectName(u"n_label")

        self.gridLayout.addWidget(self.n_label, 4, 0, 1, 1)

        self.skin_doubleSpinBox = QDoubleSpinBox(self.params_group)
        self.skin_doubleSpinBox.setObjectName(u"skin_doubleSpinBox")
        self.skin_doubleSpinBox.setReadOnly(True)

        self.gridLayout.addWidget(self.skin_doubleSpinBox, 3, 1, 1, 1)

        self.width_doubleSpinBox = QDoubleSpinBox(self.params_group)
        self.width_doubleSpinBox.setObjectName(u"width_doubleSpinBox")
        self.width_doubleSpinBox.setReadOnly(True)
        self.width_doubleSpinBox.setDecimals(2)
        self.width_doubleSpinBox.setMaximum(1000000.000000000000000)

        self.gridLayout.addWidget(self.width_doubleSpinBox, 0, 1, 1, 1)

        self.thickness_doubleSpinBox = QDoubleSpinBox(self.params_group)
        self.thickness_doubleSpinBox.setObjectName(u"thickness_doubleSpinBox")
        self.thickness_doubleSpinBox.setReadOnly(True)

        self.gridLayout.addWidget(self.thickness_doubleSpinBox, 1, 1, 1, 1)

        self.aL_label = QLabel(self.params_group)
        self.aL_label.setObjectName(u"aL_label")

        self.gridLayout.addWidget(self.aL_label, 2, 0, 1, 1)

        self.w_label = QLabel(self.params_group)
        self.w_label.setObjectName(u"w_label")

        self.gridLayout.addWidget(self.w_label, 0, 0, 1, 1)

        self.aL_doubleSpinBox = QDoubleSpinBox(self.params_group)
        self.aL_doubleSpinBox.setObjectName(u"aL_doubleSpinBox")
        self.aL_doubleSpinBox.setReadOnly(True)

        self.gridLayout.addWidget(self.aL_doubleSpinBox, 2, 1, 1, 1)


        self.verticalLayout.addWidget(self.params_group)

        self.groupBox_3 = QGroupBox(self.left_panel)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout_3 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.well_combo_dim = QComboBox(self.groupBox_3)
        self.well_combo_dim.setObjectName(u"well_combo_dim")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.well_combo_dim.sizePolicy().hasHeightForWidth())
        self.well_combo_dim.setSizePolicy(sizePolicy2)
        self.well_combo_dim.setMinimumSize(QSize(311, 0))

        self.verticalLayout_3.addWidget(self.well_combo_dim)


        self.verticalLayout.addWidget(self.groupBox_3)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.interpolar_model_group = QGroupBox(self.left_panel)
        self.interpolar_model_group.setObjectName(u"interpolar_model_group")
        self.gridLayout1 = QGridLayout(self.interpolar_model_group)
        self.gridLayout1.setObjectName(u"gridLayout1")
        self.interpolar_save_model_btn = QPushButton(self.interpolar_model_group)
        self.interpolar_save_model_btn.setObjectName(u"interpolar_save_model_btn")

        self.gridLayout1.addWidget(self.interpolar_save_model_btn, 0, 2, 1, 1)

        self.interpolar_train_model_btn = QPushButton(self.interpolar_model_group)
        self.interpolar_train_model_btn.setObjectName(u"interpolar_train_model_btn")

        self.gridLayout1.addWidget(self.interpolar_train_model_btn, 0, 0, 1, 1)

        self.interpolar_model_status_label = QLabel(self.interpolar_model_group)
        self.interpolar_model_status_label.setObjectName(u"interpolar_model_status_label")
        self.interpolar_model_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout1.addWidget(self.interpolar_model_status_label, 1, 2, 1, 1)

        self.interpolar_load_model_btn = QPushButton(self.interpolar_model_group)
        self.interpolar_load_model_btn.setObjectName(u"interpolar_load_model_btn")

        self.gridLayout1.addWidget(self.interpolar_load_model_btn, 1, 0, 1, 1)


        self.horizontalLayout_2.addWidget(self.interpolar_model_group)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.approx_model_group = QGroupBox(self.left_panel)
        self.approx_model_group.setObjectName(u"approx_model_group")
        self.gridLayout2 = QGridLayout(self.approx_model_group)
        self.gridLayout2.setObjectName(u"gridLayout2")
        self.approx_save_model_btn = QPushButton(self.approx_model_group)
        self.approx_save_model_btn.setObjectName(u"approx_save_model_btn")

        self.gridLayout2.addWidget(self.approx_save_model_btn, 0, 1, 1, 1)

        self.approx_model_status_label = QLabel(self.approx_model_group)
        self.approx_model_status_label.setObjectName(u"approx_model_status_label")
        self.approx_model_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout2.addWidget(self.approx_model_status_label, 1, 1, 1, 1)

        self.approx_train_model_btn = QPushButton(self.approx_model_group)
        self.approx_train_model_btn.setObjectName(u"approx_train_model_btn")

        self.gridLayout2.addWidget(self.approx_train_model_btn, 0, 0, 1, 1)

        self.approx_load_model_btn = QPushButton(self.approx_model_group)
        self.approx_load_model_btn.setObjectName(u"approx_load_model_btn")

        self.gridLayout2.addWidget(self.approx_load_model_btn, 1, 0, 1, 1)


        self.verticalLayout.addWidget(self.approx_model_group)

        self.report_group = QGroupBox(self.left_panel)
        self.report_group.setObjectName(u"report_group")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.report_group.sizePolicy().hasHeightForWidth())
        self.report_group.setSizePolicy(sizePolicy3)
        self.verticalLayout_2 = QVBoxLayout(self.report_group)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.text_report = QTextEdit(self.report_group)
        self.text_report.setObjectName(u"text_report")
        sizePolicy3.setHeightForWidth(self.text_report.sizePolicy().hasHeightForWidth())
        self.text_report.setSizePolicy(sizePolicy3)
        self.text_report.setMinimumSize(QSize(0, 0))
        self.text_report.setMaximumSize(QSize(16777215, 16777215))
        self.text_report.setReadOnly(True)

        self.verticalLayout_2.addWidget(self.text_report)


        self.verticalLayout.addWidget(self.report_group)


        self.horizontalLayout_3.addWidget(self.left_panel)

        self.tab_widget = QTabWidget(self.centralwidget)
        self.tab_widget.setObjectName(u"tab_widget")
        sizePolicy3.setHeightForWidth(self.tab_widget.sizePolicy().hasHeightForWidth())
        self.tab_widget.setSizePolicy(sizePolicy3)
        self.timeseries_tab = QWidget()
        self.timeseries_tab.setObjectName(u"timeseries_tab")
        self.vboxLayout = QVBoxLayout(self.timeseries_tab)
        self.vboxLayout.setObjectName(u"vboxLayout")
        self.control_gbox = QGroupBox(self.timeseries_tab)
        self.control_gbox.setObjectName(u"control_gbox")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.control_gbox.sizePolicy().hasHeightForWidth())
        self.control_gbox.setSizePolicy(sizePolicy4)
        self.control_gbox.setMinimumSize(QSize(0, 234))
        self.control_gbox.setMaximumSize(QSize(1581, 265))
        self.horizontalLayout_7 = QHBoxLayout(self.control_gbox)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.interp_btn = QPushButton(self.control_gbox)
        self.interp_btn.setObjectName(u"interp_btn")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.interp_btn.sizePolicy().hasHeightForWidth())
        self.interp_btn.setSizePolicy(sizePolicy5)

        self.verticalLayout_4.addWidget(self.interp_btn)

        self.extrapolate_btn = QPushButton(self.control_gbox)
        self.extrapolate_btn.setObjectName(u"extrapolate_btn")
        sizePolicy5.setHeightForWidth(self.extrapolate_btn.sizePolicy().hasHeightForWidth())
        self.extrapolate_btn.setSizePolicy(sizePolicy5)

        self.verticalLayout_4.addWidget(self.extrapolate_btn)

        self.ml_filter_btn = QPushButton(self.control_gbox)
        self.ml_filter_btn.setObjectName(u"ml_filter_btn")
        sizePolicy5.setHeightForWidth(self.ml_filter_btn.sizePolicy().hasHeightForWidth())
        self.ml_filter_btn.setSizePolicy(sizePolicy5)

        self.verticalLayout_4.addWidget(self.ml_filter_btn)

        self.outlier_btn = QPushButton(self.control_gbox)
        self.outlier_btn.setObjectName(u"outlier_btn")
        sizePolicy5.setHeightForWidth(self.outlier_btn.sizePolicy().hasHeightForWidth())
        self.outlier_btn.setSizePolicy(sizePolicy5)

        self.verticalLayout_4.addWidget(self.outlier_btn)

        self.fit_xy_btn = QPushButton(self.control_gbox)
        self.fit_xy_btn.setObjectName(u"fit_xy_btn")
        sizePolicy5.setHeightForWidth(self.fit_xy_btn.sizePolicy().hasHeightForWidth())
        self.fit_xy_btn.setSizePolicy(sizePolicy5)

        self.verticalLayout_4.addWidget(self.fit_xy_btn)

        self.export_btn = QPushButton(self.control_gbox)
        self.export_btn.setObjectName(u"export_btn")
        sizePolicy5.setHeightForWidth(self.export_btn.sizePolicy().hasHeightForWidth())
        self.export_btn.setSizePolicy(sizePolicy5)

        self.verticalLayout_4.addWidget(self.export_btn)

        self.reset_plots_btn = QPushButton(self.control_gbox)
        self.reset_plots_btn.setObjectName(u"reset_plots_btn")
        sizePolicy5.setHeightForWidth(self.reset_plots_btn.sizePolicy().hasHeightForWidth())
        self.reset_plots_btn.setSizePolicy(sizePolicy5)

        self.verticalLayout_4.addWidget(self.reset_plots_btn)


        self.horizontalLayout_7.addLayout(self.verticalLayout_4)

        self.groupBox_2 = QGroupBox(self.control_gbox)
        self.groupBox_2.setObjectName(u"groupBox_2")
        sizePolicy3.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy3)
        self.groupBox_2.setMinimumSize(QSize(0, 150))
        self.horizontalLayout_5 = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.p_graphic = QWidget(self.groupBox_2)
        self.p_graphic.setObjectName(u"p_graphic")
        sizePolicy3.setHeightForWidth(self.p_graphic.sizePolicy().hasHeightForWidth())
        self.p_graphic.setSizePolicy(sizePolicy3)

        self.horizontalLayout_5.addWidget(self.p_graphic)

        self.q_graphic = QWidget(self.groupBox_2)
        self.q_graphic.setObjectName(u"q_graphic")
        sizePolicy3.setHeightForWidth(self.q_graphic.sizePolicy().hasHeightForWidth())
        self.q_graphic.setSizePolicy(sizePolicy3)

        self.horizontalLayout_5.addWidget(self.q_graphic)


        self.horizontalLayout_7.addWidget(self.groupBox_2)


        self.vboxLayout.addWidget(self.control_gbox)

        self.dimensionless_gbox = QGroupBox(self.timeseries_tab)
        self.dimensionless_gbox.setObjectName(u"dimensionless_gbox")
        self.horizontalLayout_6 = QHBoxLayout(self.dimensionless_gbox)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.groupBox = QGroupBox(self.dimensionless_gbox)
        self.groupBox.setObjectName(u"groupBox")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy6)
        self.groupBox.setFlat(False)
        self.verticalLayout_5 = QVBoxLayout(self.groupBox)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.cb_dim_pD = QCheckBox(self.groupBox)
        self.cb_dim_pD.setObjectName(u"cb_dim_pD")

        self.verticalLayout_5.addWidget(self.cb_dim_pD)

        self.cb_dim_dpD = QCheckBox(self.groupBox)
        self.cb_dim_dpD.setObjectName(u"cb_dim_dpD")

        self.verticalLayout_5.addWidget(self.cb_dim_dpD)

        self.cb_dim_tD = QCheckBox(self.groupBox)
        self.cb_dim_tD.setObjectName(u"cb_dim_tD")

        self.verticalLayout_5.addWidget(self.cb_dim_tD)

        self.cb_dim_CD = QCheckBox(self.groupBox)
        self.cb_dim_CD.setObjectName(u"cb_dim_CD")

        self.verticalLayout_5.addWidget(self.cb_dim_CD)

        self.cb_XY_plot = QCheckBox(self.groupBox)
        self.cb_XY_plot.setObjectName(u"cb_XY_plot")

        self.verticalLayout_5.addWidget(self.cb_XY_plot)

        self.cb_calc_XY = QCheckBox(self.groupBox)
        self.cb_calc_XY.setObjectName(u"cb_calc_XY")

        self.verticalLayout_5.addWidget(self.cb_calc_XY)


        self.verticalLayout_7.addWidget(self.groupBox)

        self.groupBox_4 = QGroupBox(self.dimensionless_gbox)
        self.groupBox_4.setObjectName(u"groupBox_4")
        sizePolicy.setHeightForWidth(self.groupBox_4.sizePolicy().hasHeightForWidth())
        self.groupBox_4.setSizePolicy(sizePolicy)
        self.verticalLayout_6 = QVBoxLayout(self.groupBox_4)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.cb_gfunc = QCheckBox(self.groupBox_4)
        self.cb_gfunc.setObjectName(u"cb_gfunc")

        self.verticalLayout_6.addWidget(self.cb_gfunc)

        self.cb_mbt = QCheckBox(self.groupBox_4)
        self.cb_mbt.setObjectName(u"cb_mbt")

        self.verticalLayout_6.addWidget(self.cb_mbt)

        self.cb_type_gry = QCheckBox(self.groupBox_4)
        self.cb_type_gry.setObjectName(u"cb_type_gry")

        self.verticalLayout_6.addWidget(self.cb_type_gry)

        self.cb_type_cinco = QCheckBox(self.groupBox_4)
        self.cb_type_cinco.setObjectName(u"cb_type_cinco")

        self.verticalLayout_6.addWidget(self.cb_type_cinco)

        self.cb_type_valko = QCheckBox(self.groupBox_4)
        self.cb_type_valko.setObjectName(u"cb_type_valko")

        self.verticalLayout_6.addWidget(self.cb_type_valko)


        self.verticalLayout_7.addWidget(self.groupBox_4)


        self.horizontalLayout_6.addLayout(self.verticalLayout_7)

        self.dim_plot = QWidget(self.dimensionless_gbox)
        self.dim_plot.setObjectName(u"dim_plot")
        sizePolicy3.setHeightForWidth(self.dim_plot.sizePolicy().hasHeightForWidth())
        self.dim_plot.setSizePolicy(sizePolicy3)

        self.horizontalLayout_6.addWidget(self.dim_plot)


        self.vboxLayout.addWidget(self.dimensionless_gbox)

        self.tab_widget.addTab(self.timeseries_tab, "")
        self.data_tab = QWidget()
        self.data_tab.setObjectName(u"data_tab")
        self.verticalLayout_8 = QVBoxLayout(self.data_tab)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.data_table = QTableView(self.data_tab)
        self.data_table.setObjectName(u"data_table")

        self.verticalLayout_8.addWidget(self.data_table)

        self.data_info_label = QLabel(self.data_tab)
        self.data_info_label.setObjectName(u"data_info_label")
        sizePolicy5.setHeightForWidth(self.data_info_label.sizePolicy().hasHeightForWidth())
        self.data_info_label.setSizePolicy(sizePolicy5)

        self.verticalLayout_8.addWidget(self.data_info_label)

        self.tab_widget.addTab(self.data_tab, "")
        self.grp_tab = QWidget()
        self.grp_tab.setObjectName(u"grp_tab")
        self.vboxLayout1 = QVBoxLayout(self.grp_tab)
        self.vboxLayout1.setObjectName(u"vboxLayout1")
        self.groupBox_6 = QGroupBox(self.grp_tab)
        self.groupBox_6.setObjectName(u"groupBox_6")
        sizePolicy4.setHeightForWidth(self.groupBox_6.sizePolicy().hasHeightForWidth())
        self.groupBox_6.setSizePolicy(sizePolicy4)
        self.horizontalLayout_10 = QHBoxLayout(self.groupBox_6)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.verticalLayout_10 = QVBoxLayout()
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.fractures_value_label = QLabel(self.groupBox_6)
        self.fractures_value_label.setObjectName(u"fractures_value_label")

        self.verticalLayout_10.addWidget(self.fractures_value_label)

        self.fracture_length_label = QLabel(self.groupBox_6)
        self.fracture_length_label.setObjectName(u"fracture_length_label")

        self.verticalLayout_10.addWidget(self.fracture_length_label)

        self.fracture_width_label = QLabel(self.groupBox_6)
        self.fracture_width_label.setObjectName(u"fracture_width_label")

        self.verticalLayout_10.addWidget(self.fracture_width_label)


        self.horizontalLayout_10.addLayout(self.verticalLayout_10)

        self.verticalLayout_11 = QVBoxLayout()
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.skin_value_label = QLabel(self.groupBox_6)
        self.skin_value_label.setObjectName(u"skin_value_label")

        self.verticalLayout_11.addWidget(self.skin_value_label)

        self.al_ratio_label = QLabel(self.groupBox_6)
        self.al_ratio_label.setObjectName(u"al_ratio_label")

        self.verticalLayout_11.addWidget(self.al_ratio_label)

        self.thickness_value_label = QLabel(self.groupBox_6)
        self.thickness_value_label.setObjectName(u"thickness_value_label")

        self.verticalLayout_11.addWidget(self.thickness_value_label)


        self.horizontalLayout_10.addLayout(self.verticalLayout_11)


        self.vboxLayout1.addWidget(self.groupBox_6)

        self.groupBox_5 = QGroupBox(self.grp_tab)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.verticalLayout_12 = QVBoxLayout(self.groupBox_5)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.flow_regime_btn = QPushButton(self.groupBox_5)
        self.flow_regime_btn.setObjectName(u"flow_regime_btn")

        self.horizontalLayout_9.addWidget(self.flow_regime_btn)

        self.transitions_btn = QPushButton(self.groupBox_5)
        self.transitions_btn.setObjectName(u"transitions_btn")

        self.horizontalLayout_9.addWidget(self.transitions_btn)

        self.productivity_btn = QPushButton(self.groupBox_5)
        self.productivity_btn.setObjectName(u"productivity_btn")

        self.horizontalLayout_9.addWidget(self.productivity_btn)


        self.verticalLayout_12.addLayout(self.horizontalLayout_9)

        self.grp_results_text = QTextEdit(self.groupBox_5)
        self.grp_results_text.setObjectName(u"grp_results_text")

        self.verticalLayout_12.addWidget(self.grp_results_text)


        self.vboxLayout1.addWidget(self.groupBox_5)

        self.tab_widget.addTab(self.grp_tab, "")
        self.type_curves_tab = QWidget()
        self.type_curves_tab.setObjectName(u"type_curves_tab")
        self.verticalLayout_9 = QVBoxLayout(self.type_curves_tab)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.linear_btn = QPushButton(self.type_curves_tab)
        self.linear_btn.setObjectName(u"linear_btn")

        self.horizontalLayout_4.addWidget(self.linear_btn)

        self.bilinear_btn = QPushButton(self.type_curves_tab)
        self.bilinear_btn.setObjectName(u"bilinear_btn")

        self.horizontalLayout_4.addWidget(self.bilinear_btn)

        self.pseudoradial_btn = QPushButton(self.type_curves_tab)
        self.pseudoradial_btn.setObjectName(u"pseudoradial_btn")

        self.horizontalLayout_4.addWidget(self.pseudoradial_btn)

        self.match_curves_btn = QPushButton(self.type_curves_tab)
        self.match_curves_btn.setObjectName(u"match_curves_btn")

        self.horizontalLayout_4.addWidget(self.match_curves_btn)


        self.verticalLayout_9.addLayout(self.horizontalLayout_4)

        self.type_curves_plot_placeholder = QWidget(self.type_curves_tab)
        self.type_curves_plot_placeholder.setObjectName(u"type_curves_plot_placeholder")
        sizePolicy3.setHeightForWidth(self.type_curves_plot_placeholder.sizePolicy().hasHeightForWidth())
        self.type_curves_plot_placeholder.setSizePolicy(sizePolicy3)

        self.verticalLayout_9.addWidget(self.type_curves_plot_placeholder)

        self.tab_widget.addTab(self.type_curves_tab, "")
        self.results_tab = QWidget()
        self.results_tab.setObjectName(u"results_tab")
        self.vboxLayout2 = QVBoxLayout(self.results_tab)
        self.vboxLayout2.setObjectName(u"vboxLayout2")
        self.results_text = QTextEdit(self.results_tab)
        self.results_text.setObjectName(u"results_text")
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.results_text.sizePolicy().hasHeightForWidth())
        self.results_text.setSizePolicy(sizePolicy7)

        self.vboxLayout2.addWidget(self.results_text)

        self.export_report_btn = QPushButton(self.results_tab)
        self.export_report_btn.setObjectName(u"export_report_btn")

        self.vboxLayout2.addWidget(self.export_report_btn)

        self.tab_widget.addTab(self.results_tab, "")

        self.horizontalLayout_3.addWidget(self.tab_widget)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.tab_widget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.left_panel.setTitle("")
        self.load_template_button.setText(QCoreApplication.translate("MainWindow", u"\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u044c csv/pq \u0444\u0430\u0439\u043b", None))
        self.load_file_label.setText(QCoreApplication.translate("MainWindow", u"\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u0435 \u0444\u0430\u0439\u043b", None))
        self.params_group.setTitle(QCoreApplication.translate("MainWindow", u"\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b", None))
        self.thickness_label.setText(QCoreApplication.translate("MainWindow", u"\u0422\u043e\u043b\u0449\u0438\u043d\u0430 \u043f\u043b\u0430\u0441\u0442\u0430 h, \u043c", None))
        self.skin_label.setText(QCoreApplication.translate("MainWindow", u"Skin", None))
        self.n_label.setText(QCoreApplication.translate("MainWindow", u"N (\u0442\u0440\u0435\u0449\u0438\u043d\u044b)", None))
        self.aL_label.setText(QCoreApplication.translate("MainWindow", u"a/L", None))
        self.w_label.setText(QCoreApplication.translate("MainWindow", u"\u0414\u043b\u0438\u043d\u0430 \u0442\u0440\u0435\u0449\u0438\u043d\u044b W, \u043c", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"\u0412\u044b\u0431\u043e\u0440 \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u044b", None))
        self.interpolar_model_group.setTitle(QCoreApplication.translate("MainWindow", u"\u041c\u043e\u0434\u0435\u043b\u044c \u0438\u043d\u0442\u0435\u0440\u043f\u043e\u043b\u044f\u0446\u0438\u0438", None))
        self.interpolar_save_model_btn.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c", None))
        self.interpolar_train_model_btn.setText(QCoreApplication.translate("MainWindow", u"\u041e\u0431\u0443\u0447\u0438\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c", None))
        self.interpolar_model_status_label.setText(QCoreApplication.translate("MainWindow", u"\u041c\u043e\u0434\u0435\u043b\u044c \u043d\u0435 \u043e\u0431\u0443\u0447\u0435\u043d\u0430", None))
        self.interpolar_load_model_btn.setText(QCoreApplication.translate("MainWindow", u"\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c", None))
        self.approx_model_group.setTitle(QCoreApplication.translate("MainWindow", u"\u041c\u043e\u0434\u0435\u043b\u044c \u0430\u043f\u043f\u0440\u043e\u043a\u0441\u0438\u043c\u0430\u0446\u0438\u0438", None))
        self.approx_save_model_btn.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c", None))
        self.approx_model_status_label.setText(QCoreApplication.translate("MainWindow", u"\u041c\u043e\u0434\u0435\u043b\u044c \u043d\u0435 \u043e\u0431\u0443\u0447\u0435\u043d\u0430", None))
        self.approx_train_model_btn.setText(QCoreApplication.translate("MainWindow", u"\u041e\u0431\u0443\u0447\u0438\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c", None))
        self.approx_load_model_btn.setText(QCoreApplication.translate("MainWindow", u"\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c", None))
        self.report_group.setTitle(QCoreApplication.translate("MainWindow", u"\u041e\u0442\u0447\u0451\u0442", None))
        self.control_gbox.setTitle(QCoreApplication.translate("MainWindow", u"\u0423\u043f\u0440\u0430\u0432\u043b\u0435\u043d\u0438\u0435", None))
        self.interp_btn.setText(QCoreApplication.translate("MainWindow", u"\u0418\u043d\u0442\u0435\u0440\u043f\u043e\u043b\u0438\u0440\u043e\u0432\u0430\u0442\u044c \u043f\u0440\u043e\u043f\u0443\u0441\u043a\u0438", None))
        self.extrapolate_btn.setText(QCoreApplication.translate("MainWindow", u"\u042d\u043a\u0441\u0442\u0440\u0430\u043f\u043e\u043b\u0438\u0440\u043e\u0432\u0430\u0442\u044c  X-Y", None))
        self.ml_filter_btn.setText(QCoreApplication.translate("MainWindow", u"ML \u0444\u0438\u043b\u044c\u0442\u0440\u0430\u0446\u0438\u044f", None))
        self.outlier_btn.setText(QCoreApplication.translate("MainWindow", u"\u041e\u0431\u043d\u0430\u0440\u0443\u0436\u0438\u0442\u044c \u0432\u044b\u0431\u0440\u043e\u0441\u044b", None))
        self.fit_xy_btn.setText(QCoreApplication.translate("MainWindow", u"\u0410\u043f\u043f\u0440\u043e\u043a\u0441\u0438\u043c\u0438\u0440\u043e\u0432\u0430\u0442\u044c X-Y", None))
        self.export_btn.setText(QCoreApplication.translate("MainWindow", u"\u042d\u043a\u0441\u043f\u043e\u0440\u0442\u0438\u0440\u043e\u0432\u0430\u0442\u044c \u0434\u0430\u043d\u043d\u044b\u0435", None))
        self.reset_plots_btn.setText(QCoreApplication.translate("MainWindow", u"\u0421\u0431\u0440\u043e\u0441\u0438\u0442\u044c \u0433\u0440\u0430\u0444\u0438\u043a\u0438", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"\u0420\u0435\u0430\u043b\u044c\u043d\u044b\u0435 \u0434\u0430\u043d\u043d\u044b\u0435", None))
        self.dimensionless_gbox.setTitle(QCoreApplication.translate("MainWindow", u"\u0420\u0430\u0441\u0447\u0435\u0442\u043d\u044b\u0435 \u0434\u0430\u043d\u043d\u044b\u0435", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"\u0411\u0435\u0437\u0440\u0430\u0437\u043c\u0435\u0440\u043d\u044b\u0435 \u0433\u0440\u0430\u0444\u0438\u043a\u0438 (log-log)", None))
        self.cb_dim_pD.setText(QCoreApplication.translate("MainWindow", u"pD(Y)", None))
        self.cb_dim_dpD.setText(QCoreApplication.translate("MainWindow", u"dp(D)/dlog(Y)", None))
        self.cb_dim_tD.setText(QCoreApplication.translate("MainWindow", u"tD(~Y)", None))
        self.cb_dim_CD.setText(QCoreApplication.translate("MainWindow", u"CD(\u0415\u043c\u043a\u043e\u0441\u0442\u044c)", None))
        self.cb_XY_plot.setText(QCoreApplication.translate("MainWindow", u"\u0418\u0441\u0445\u043e\u0434\u043d\u044b\u0435 X-Y", None))
        self.cb_calc_XY.setText(QCoreApplication.translate("MainWindow", u"\u0420\u0430\u0441\u0447\u0435\u0442\u043d\u044b\u0435 X-Y", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"\u0422\u0438\u043f\u043e\u0432\u044b\u0435 \u043a\u0440\u0438\u0432\u044b\u0435", None))
        self.cb_gfunc.setText(QCoreApplication.translate("MainWindow", u"G-\u0444\u0443\u043d\u043a\u0446\u0438\u044f(Nolte)", None))
        self.cb_mbt.setText(QCoreApplication.translate("MainWindow", u"\u0412\u0440\u0435\u043c\u044f \u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b\u044c\u043d\u043e\u0433\u043e \u0431\u0430\u043b\u0430\u043d\u0441\u0430", None))
        self.cb_type_gry.setText(QCoreApplication.translate("MainWindow", u"\u041b\u0438\u043d\u0435\u0439\u043d\u044b\u0439 \u0440\u0435\u0436\u0438\u043c", None))
        self.cb_type_cinco.setText(QCoreApplication.translate("MainWindow", u"\u0411\u0438\u043b\u0438\u043d\u0435\u0439\u043d\u044b\u0439 \u0440\u0435\u0436\u0438\u043c", None))
        self.cb_type_valko.setText(QCoreApplication.translate("MainWindow", u"\u041f\u0441\u0435\u0432\u0434\u043e\u0440\u0430\u0434\u0438\u0430\u043b\u044c\u043d\u044b\u0439 \u0440\u0435\u0436\u0438\u043c", None))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.timeseries_tab), QCoreApplication.translate("MainWindow", u"\u041e\u043f\u0435\u0440\u0430\u0446\u0438\u0438 \u0438 \u0433\u0440\u0430\u0444\u0438\u043a\u0438", None))
        self.data_info_label.setText(QCoreApplication.translate("MainWindow", u"\u0414\u0430\u043d\u043d\u044b\u0435 \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u044b \u0435\u0449\u0435 \u043d\u0435 \u0437\u0430\u0433\u0440\u0443\u0436\u0435\u043d\u044b", None))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.data_tab), QCoreApplication.translate("MainWindow", u"\u0422\u0430\u0431\u043b\u0438\u0447\u043d\u043e\u0435 \u043f\u0440\u0435\u0434\u0441\u0442\u0430\u0432\u043b\u0435\u043d\u0438\u0435", None))
        self.groupBox_6.setTitle(QCoreApplication.translate("MainWindow", u"\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b \u0413\u0420\u041f", None))
        self.fractures_value_label.setText(QCoreApplication.translate("MainWindow", u"\u0422\u0440\u0435\u0449\u0438\u043d\u044b: -", None))
        self.fracture_length_label.setText(QCoreApplication.translate("MainWindow", u"\u0414\u043b\u0438\u043d\u0430 \u0442\u0440\u0435\u0449\u0438\u043d\u044b: -", None))
        self.fracture_width_label.setText(QCoreApplication.translate("MainWindow", u"\u0428\u0438\u0440\u0438\u043d\u0430 \u0442\u0440\u0435\u0449\u0438\u043d\u044b: -", None))
        self.skin_value_label.setText(QCoreApplication.translate("MainWindow", u"Skin: -", None))
        self.al_ratio_label.setText(QCoreApplication.translate("MainWindow", u"a/L: -", None))
        self.thickness_value_label.setText(QCoreApplication.translate("MainWindow", u"\u0422\u043e\u043b\u0449\u0438\u043d\u0430: -", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("MainWindow", u"\u0410\u043d\u0430\u043b\u0438\u0437", None))
        self.flow_regime_btn.setText(QCoreApplication.translate("MainWindow", u"\u0410\u043d\u0430\u043b\u0438\u0437 \u0440\u0435\u0436\u0438\u043c\u0430 \u0442\u0435\u0447\u0435\u043d\u0438\u044f", None))
        self.transitions_btn.setText(QCoreApplication.translate("MainWindow", u"\u0418\u043d\u0434\u0435\u043a\u0441 \u043f\u0440\u043e\u0434\u0443\u043a\u0442\u0438\u0432\u043d\u043e\u0441\u0442\u0438", None))
        self.productivity_btn.setText(QCoreApplication.translate("MainWindow", u"\u041f\u0435\u0440\u0435\u0445\u043e\u0434\u044b \u0440\u0435\u0436\u0438\u043c\u043e\u0432", None))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.grp_tab), QCoreApplication.translate("MainWindow", u"\u0410\u043d\u0430\u043b\u0438\u0437 \u0413\u0420\u041f", None))
        self.linear_btn.setText(QCoreApplication.translate("MainWindow", u"\u041b\u0438\u043d\u0435\u0439\u043d\u043e\u0435 \u0442\u0435\u0447\u0435\u043d\u0438\u0435", None))
        self.bilinear_btn.setText(QCoreApplication.translate("MainWindow", u"\u0411\u0438\u043b\u0438\u043d\u0435\u0439\u043d\u043e\u0435 \u0442\u0435\u0447\u0435\u043d\u0438\u0435", None))
        self.pseudoradial_btn.setText(QCoreApplication.translate("MainWindow", u"\u041f\u0441\u0435\u0432\u0434\u043e\u0440\u0430\u0434\u0438\u0430\u043b\u044c\u043d\u043e\u0435 \u0442\u0435\u0447\u0435\u043d\u0438\u0435", None))
        self.match_curves_btn.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u043f\u043e\u0441\u0442\u0430\u0432\u0438\u0442\u044c \u0441 \u0434\u0430\u043d\u043d\u044b\u043c\u0438", None))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.type_curves_tab), QCoreApplication.translate("MainWindow", u"\u042d\u0442\u0430\u043b\u043e\u043d\u043d\u044b\u0435 \u043a\u0440\u0438\u0432\u044b\u0435", None))
        self.export_report_btn.setText(QCoreApplication.translate("MainWindow", u"\u042d\u043a\u0441\u043f\u043e\u0440\u0442 \u043e\u0442\u0447\u0435\u0442\u0430", None))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.results_tab), QCoreApplication.translate("MainWindow", u"\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b", None))
    # retranslateUi

