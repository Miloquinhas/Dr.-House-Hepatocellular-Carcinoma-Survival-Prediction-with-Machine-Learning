import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QTextEdit, QGridLayout, QFrame, QScrollArea
)
from PyQt5.QtGui import QFont, QLinearGradient, QColor, QPainter, QFontDatabase, QIcon, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class GradientWidget(QWidget):
    def __init__(self):
        super().__init__()

    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor("#e0f7fa"))
        gradient.setColorAt(1.0, QColor("#e0f2f1"))
        painter.fillRect(self.rect(), gradient)

class DrHouseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.variables = {}  # Initialize self.variables before using it
        self.fields = [
            "Gender", "Symptoms", "Alcohol", "HBsAg", "HBeAg", "HBcAb", "HCVAb", 
            "Cirrhosis", "Endemic", "Smoking", "Diabetes", "Obesity", "Hemochro", 
            "AHT", "CRI", "HIV", "NASH", "Varices", "Spleno", "PHT", "PVT", 
            "Metastasis", "Hallmark", "Age", "Grams_day", "Packs_year", "PS", 
            "Encephalopathy", "Ascites", "INR", "AFP", "Hemoglobin", "MCV", 
            "Leucocytes", "Platelets", "Albumin", "Total_Bil", "ALT", "AST", 
            "GGT", "ALP", "TP", "Creatinine", "Nodules", "Major_Dim", "Dir_Bil", 
            "Iron", "Sat", "Ferritin"
        ]
        
        self.initUI()
        self.load_or_train_model()  # Call this after initializing UI

        # Initialize media players for sounds
        self.happy_player = QMediaPlayer()
        self.happy_player.setMedia(QMediaContent(QUrl.fromLocalFile("sound/happy.wav")))
        self.happy_player.setVolume(40)  # Set volume to 40%

        self.womp_womp_player = QMediaPlayer()
        self.womp_womp_player.setMedia(QMediaContent(QUrl.fromLocalFile("sound/womp_womp.wav")))
        self.womp_womp_player.setVolume(40)  # Set volume to 40%

    def initUI(self):
        self.setWindowTitle('Dr. House')
        self.setGeometry(100, 100, 700, 600)  # Adjusted window size

        # Load custom font
        font_db = QFontDatabase()
        font_id = font_db.addApplicationFont("font/PoetsenOne-Regular.ttf")
        self.font_id = font_id  # Store font_id as an instance attribute
        if font_id != -1:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
        else:
            font_family = self.font().family()

        central_widget = GradientWidget()
        central_widget.setObjectName("centralWidget")  # Set object name for styling

        # Create a frame for the border
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame.setStyleSheet("""
            QFrame {
                border: 5px solid navy;  # Increased border thickness
                border-radius: 10px;
            }
        """)

        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(central_widget)

        self.setCentralWidget(frame)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Title Bar with Icons
        title_layout = QHBoxLayout()
        
        title_icon = QLabel()
        title_icon.setPixmap(QIcon("icons/stethoscope.png").pixmap(30, 24))  # Add stethoscope icon
        title_layout.addWidget(title_icon, alignment=Qt.AlignLeft)
        
        title = QLabel("Dr. House")
        title.setFont(QFont(font_family, 18, QFont.Bold))  # Adjusted font size
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title, alignment=Qt.AlignCenter)
        
        lab_test_icon = QLabel()
        lab_test_icon.setPixmap(QIcon("icons/lab_test.png").pixmap(30, 30))  # Add lab test icon
        title_layout.addWidget(lab_test_icon, alignment=Qt.AlignRight)
        
        main_layout.addLayout(title_layout)

        # Grid layout for input fields
        grid_layout = QGridLayout()
        grid_layout.setSpacing(8)  # Adjusted spacing

        self.ranges = {
            "AFP": (0, 2624.0),
            "Hemoglobin": (6.23, 19.869999999999997),
            "MCV": (71.91, 118.99000000000001),
            "Leucocytes": (0, 111.75),
            "Platelets": (0, 458541.0),
            "Albumin": (1.5700000000000003, 5.529999999999999),
            "Total_Bil": (0, 6.8074312500000005),
            "ALT": (0, 155.2),
            "AST": (0, 211.7),
            "GGT": (0, 769.7),
            "ALP": (0, 516.7),
            "TP": (4.190000000000001, 9.91),
            "Creatinine": (0.0009493670886076, 1.836139240506329),
            "Nodules": (0, 11.8),
            "Major_Dim": (0, 17.85),
            "Dir_Bil": (0, 4.361000000000001),
            "Iron": (69.7, 109.3),
            "Sat": (20.0, 64.0),
            "Ferritin": (91.19999999999999, 988.8),
            "Grams_day": (0, 270.0),
            "Packs_year": (0, 54.0),
            "INR": (0.5849999999999999, 2.125)
        }

        row, col = 0, 0
        for field in self.fields:
            label = QLabel(f"{field}:")
            label.setFont(QFont(font_family, 10))  # Adjusted font size

            if field in [
                "Gender", "Symptoms", "Alcohol", "HBsAg", "HBeAg", "HBcAb", "HCVAb", 
                "Cirrhosis", "Endemic", "Smoking", "Diabetes", "Obesity", "Hemochro", 
                "AHT", "CRI", "HIV", "NASH", "Varices", "Spleno", "PHT", "PVT", 
                "Metastasis", "Hallmark", "PS", "Encephalopathy", "Ascites"
            ]:
                combo = QComboBox()
                combo.setFont(QFont(font_family, 10))  # Adjusted font size
                if field == "Gender":
                    options = ["Male", "Female"]
                elif field in ["Alcohol", "Cirrhosis"]:
                    options = ["Yes", "No"]
                elif field == "PS":
                    options = ["Active", "Selfcare", "Ambulatory", "Disabled"]
                elif field == "Encephalopathy":
                    options = ["Grade I/II", "Grade III/IV", "None"]
                elif field == "Ascites":
                    options = ["Mild", "Moderate/Severe", "None"]
                else:
                    options = ["Yes", "No", "?"]
                combo.addItems(options)
                combo.setStyleSheet("""
                    QComboBox {
                        padding: 5px;
                        border: 3px solid navy;  # Apply navy blue border to combobox
                        border-radius: 5px;
                        background-color: #ffffff;
                        box-shadow: inset 1px 1px 5px rgba(0,0,0,0.1);
                    }
                                        QComboBox::drop-down {
                        border: none;
                    }
                """)
                grid_layout.addWidget(label, row, col)
                grid_layout.addWidget(combo, row, col + 1)
                self.variables[field] = combo
            else:
                line_edit = QLineEdit()
                line_edit.setFont(QFont(font_family, 10))  # Adjusted font size
                line_edit.setStyleSheet("""
                    QLineEdit {
                        padding: 5px;
                        border: 3px solid navy;  # Apply navy blue border to line edit
                        border-radius: 5px;
                        background-color: #ffffff;
                        box-shadow: inset 1px 1px 5px rgba(0,0,0,0.1);
                    }
                """)
                grid_layout.addWidget(label, row, col)
                grid_layout.addWidget(line_edit, row, col + 1)
                self.variables[field] = line_edit

            col += 2
            if col >= 6:
                col = 0
                row += 1

        main_layout.addLayout(grid_layout)

        # Button Layout
        button_layout = QHBoxLayout()

        # Show Ranges Button
        ranges_button = QPushButton('Show Ranges')
        ranges_button.setFont(QFont(font_family, 12, QFont.Bold))  # Adjusted font size
        ranges_button.setIcon(QIcon("icons/ranges.png"))  # Add ranges icon
        ranges_button.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                padding: 10px;
                border-radius: 10px;
                border: 3px solid navy;  # Apply navy blue border to button
            }
            QPushButton:hover {
                background-color: #005bb5;
            }
        """)
        ranges_button.clicked.connect(self.show_ranges)
        button_layout.addWidget(ranges_button)

        # Submit Button
        submit_button = QPushButton('Submit')
        submit_button.setFont(QFont(font_family, 12, QFont.Bold))  # Adjusted font size
        submit_button.setIcon(QIcon("icons/submit_icon.png"))  # Add submit icon
        submit_button.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                padding: 10px;
                border-radius: 10px;
                border: 3px solid navy;  # Apply navy blue border to button
            }
            QPushButton:hover {
                background-color: #005bb5;
            }
        """)
        submit_button.clicked.connect(self.show_selections)
        button_layout.addWidget(submit_button)

        # Graph Button
        graph_button = QPushButton('Show Graphs')
        graph_button.setFont(QFont(font_family, 12, QFont.Bold))  # Adjusted font size
        graph_button.setIcon(QIcon("icons/graph_icon.png"))  # Add graph icon
        graph_button.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                padding: 10px;
                border-radius: 10px;
                border: 3px solid navy;  # Apply navy blue border to button
            }
            QPushButton:hover {
                background-color: #005bb5;
            }
        """)
        graph_button.clicked.connect(self.show_graphs)
        button_layout.addWidget(graph_button)

        main_layout.addLayout(button_layout)

        # Result Text Edit
        self.result_text = QTextEdit()
        self.result_text.setFont(QFont(font_family, 10))  # Adjusted font size
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 3px solid navy;  # Apply navy blue border to text edit
                padding: 10px;
                border-radius: 10px;
                box-shadow: inset 1px 1px 5px rgba(0,0,0,0.1);
            }
        """)
        self.result_text.setReadOnly(True)
        main_layout.addWidget(self.result_text)

    def load_or_train_model(self):
        # Load the dataset
        patient_data = pd.read_csv(r"data/hcc_dataset.csv")

        # Prepare the data
        data_for_test = patient_data.copy()
        input_features = data_for_test.columns[:-1]
        target_feature = 'Class'

        x_bench = data_for_test[input_features]
        y_bench = data_for_test[target_feature]

        x_encoded = pd.get_dummies(x_bench)
        y_encoded = pd.get_dummies(y_bench)

        x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, test_size=0.33, random_state=20)

        if y_train.shape[1] == 1:
            y_train_e = y_train.squeeze()
        else:
            y_train_e = y_train.idxmax(axis=1)
        if y_test.shape[1] == 1:
            y_test_e = y_test.squeeze()
        else:
            y_test_e = y_test.idxmax(axis=1)

        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model.fit(x_train, y_train_e)
        self.columns_encoded = x_encoded.columns

        # Optionally, evaluate the model
        y_pred = self.model.predict(x_test)
        print(classification_report(y_test_e, y_pred))



    def preprocess_input(self):
        res = ""
        for field in self.fields:  # Use the ordered fields list
            widget = self.variables[field]
            value = widget.currentText() if isinstance(widget, QComboBox) else widget.text()
            
            if field in self.ranges:
                try:
                    num_value = float(value)
                    if not (self.ranges[field][0] <= num_value <= self.ranges[field][1]):
                        self.result_text.setText("Value Not Accepted")
                        return None
                except ValueError:
                    self.result_text.setText("Value Not Accepted")
                    return None
            res += f"{value}, "

        new_data = res.split(', ')
        if new_data[-1] == "":
            new_data = new_data[:-1]  # Remove the trailing empty element
        new_data_df = pd.DataFrame([new_data], columns=self.fields)  # Use the ordered fields list
        df_encoded = pd.get_dummies(new_data_df)
        
        missing_cols = set(self.columns_encoded) - set(df_encoded.columns)
        for col in missing_cols:
            df_encoded[col] = 0

        return df_encoded[self.columns_encoded]

    def predict_outcome(self, data):
        prediction = self.model.predict(data)
        if prediction[0] == "Lives":
            self.happy_player.play()
            return "Patient Will Survive"
        else:
            self.womp_womp_player.play()
            return "Patient Will NOT Survive"

    def show_selections(self):
        res = ""
        count = 0
        for key, widget in self.variables.items():
            value = widget.currentText() if isinstance(widget, QComboBox) else widget.text()
            
            res += f"{value}, "
            count += 1
            if count % 6 == 0:
                res += "\n"

        new_data = res.split(', ')
        if new_data[-1] == "":
            new_data = new_data[:-1]  # Remove the trailing empty element
        new_data_df = pd.DataFrame([new_data], columns=self.fields)  # Use the ordered fields list
        df_encoded = pd.get_dummies(new_data_df)
        
        missing_cols = set(self.columns_encoded) - set(df_encoded.columns)
        for col in missing_cols:
            df_encoded[col] = 0

        df_encoded = df_encoded[self.columns_encoded]

        pred = self.model.predict(df_encoded)
        
        if pred[0] == "Lives":
            self.happy_player.play()
            self.result_text.setText("Patient Will Survive")
        else:
            self.womp_womp_player.play()
            self.result_text.setText("Patient Will NOT Survive")


    def show_graphs(self):
        graph_window = QMainWindow(self)
        graph_window.setWindowTitle("Graphs")
        graph_window.setGeometry(150, 150, 1000, 800)  # Adjusted window size

        central_widget = GradientWidget()
        graph_window.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        graphs_folder = 'graphs'
        graph_files = [
            "Screenshot_2024-05-20_233849.png",
            "Screenshot_2024-05-20_233905.png",
            "Screenshot_2024-05-21_002247.png",
            "Screenshot_2024-05-21_002303.png",
            "Screenshot_2024-05-21_002737.png",
            "Screenshot_2024-05-21_002951.png"
        ]

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        for graph_file in graph_files:
            graph_path = os.path.join(graphs_folder, graph_file)
            if os.path.exists(graph_path):
                graph_label = QLabel()
                pixmap = QPixmap(graph_path).scaled(800, 600, Qt.KeepAspectRatio)  # Resize images to fit
                graph_label.setPixmap(pixmap)
                graph_label.setAlignment(Qt.AlignCenter)
                scroll_layout.addWidget(graph_label)

        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        graph_window.show()


    def show_ranges(self):
        ranges_window = QMainWindow(self)
        ranges_window.setWindowTitle("Accepted Ranges")
        ranges_window.setGeometry(150, 150, 800, 600)  # Adjusted window size

        central_widget = GradientWidget()
        ranges_window.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        for var, (low, high) in self.ranges.items():
            range_label = QLabel(f"{var}:   {low} -> {high}")
            range_label.setFont(QFont(QFontDatabase.applicationFontFamilies(self.font_id)[0], 12))
            range_label.setAlignment(Qt.AlignLeft)
            scroll_layout.addWidget(range_label)

        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        ranges_window.show()


def main():
    app = QApplication(sys.argv)
    ex = DrHouseApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

                       
