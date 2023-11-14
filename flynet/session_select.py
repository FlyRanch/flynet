# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'session_select.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Session_Dialog(object):
    def setupUi(self, Session_Dialog):
        Session_Dialog.setObjectName("Session_Dialog")
        Session_Dialog.resize(400, 300)
        self.gridLayout = QtWidgets.QGridLayout(Session_Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.folder_tree = QtWidgets.QTreeView(Session_Dialog)
        self.folder_tree.setObjectName("folder_tree")
        self.gridLayout.addWidget(self.folder_tree, 0, 0, 1, 1)
        self.selected_session = QtWidgets.QLineEdit(Session_Dialog)
        self.selected_session.setObjectName("selected_session")
        self.gridLayout.addWidget(self.selected_session, 1, 0, 1, 1)
        self.session_select_window = QtWidgets.QDialogButtonBox(Session_Dialog)
        self.session_select_window.setOrientation(QtCore.Qt.Horizontal)
        self.session_select_window.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.session_select_window.setObjectName("session_select_window")
        self.gridLayout.addWidget(self.session_select_window, 2, 0, 1, 1)

        self.retranslateUi(Session_Dialog)
        self.session_select_window.accepted.connect(Session_Dialog.accept)
        self.session_select_window.rejected.connect(Session_Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Session_Dialog)

    def retranslateUi(self, Session_Dialog):
        _translate = QtCore.QCoreApplication.translate
        Session_Dialog.setWindowTitle(_translate("Session_Dialog", "Select session folder"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Session_Dialog = QtWidgets.QDialog()
    ui = Ui_Session_Dialog()
    ui.setupUi(Session_Dialog)
    Session_Dialog.show()
    sys.exit(app.exec_())

