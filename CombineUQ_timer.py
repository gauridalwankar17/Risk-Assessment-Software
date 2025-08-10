import sys
import time
import numpy as np
import numexpr as ne
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel,QFileDialog,
    QTableWidget, QTableWidgetItem, QMessageBox, QLineEdit, QComboBox, QListWidget, QHBoxLayout, QWidget, QGroupBox
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from pyDOE import lhs
from SALib.sample import saltelli
from sklearn.cluster import KMeans
import os
import pandas as pd
import win32com.client
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, 
    QTableWidget, QTableWidgetItem, QMessageBox, QLineEdit, QComboBox, QHBoxLayout
)
from PyQt5.QtCore import Qt
from pyDOE import lhs

# Sampling Methods
def monte_carlo_sample(bounds, n_samples):
    num_vars = len(bounds)
    samples = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(n_samples, num_vars)
    )
    return samples

def quasi_monte_carlo_sample(bounds, n_samples):
    num_vars = len(bounds)
    samples = np.random.random((n_samples, num_vars))
    samples = samples * (np.array([b[1] - b[0] for b in bounds])) + np.array([b[0] for b in bounds])
    return samples

def latin_hypercube_sample(bounds, n_samples):
    num_vars = len(bounds)
    lhs_samples = lhs(num_vars, samples=n_samples)
    samples = lhs_samples * (np.array([b[1] - b[0] for b in bounds])) + np.array([b[0] for b in bounds])
    return samples

def orthogonal_array_sample(bounds, n_samples):
    num_vars = len(bounds)
    step = n_samples // num_vars
    samples = np.array([np.linspace(b[0], b[1], step) for b in bounds]).T
    return samples

def gradient_sample(bounds, n_samples):
    num_vars = len(bounds)
    base_sample = monte_carlo_sample(bounds, n_samples)
    gradient_direction = np.random.randn(n_samples, num_vars)
    samples = base_sample + gradient_direction * 0.1
    return samples

def metis_sample(bounds, n_samples, n_clusters=3):
    base_sample = monte_carlo_sample(bounds, n_samples)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(base_sample)
    return kmeans.cluster_centers_

def sobol_sample(bounds, n_samples):
    problem = {
        "num_vars": len(bounds),
        "names": [f"x{i}" for i in range(len(bounds))],
        "bounds": bounds,
    }
    adjusted_samples = 2 ** int(np.ceil(np.log2(n_samples)))
    samples = saltelli.sample(problem, adjusted_samples)
    return samples[:n_samples], problem

def morris_design(bounds, n_samples):
    step_size = 0.1
    base_sample = monte_carlo_sample(bounds, n_samples)
    perturbed_sample = base_sample + step_size
    return np.vstack([base_sample, perturbed_sample])

def generalized_morris_design(bounds, n_samples, num_levels=4):
    num_vars = len(bounds)
    levels = np.linspace(0, 1, num_levels)
    grid = np.array(np.meshgrid(*[levels for _ in range(num_vars)])).T.reshape(-1, num_vars)
    samples = grid * (np.array([b[1] - b[0] for b in bounds])) + np.array([b[0] for b in bounds])
    return samples[:n_samples]

# Plot Functions
def plot_output_distribution(outputs, variable_name):
    plt.figure(figsize=(10, 5))
    plt.hist(outputs, bins=15, edgecolor="black", alpha=0.7)
    plt.title(f"Probability Distribution for {variable_name}")
    plt.xlabel(variable_name)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_1d_scatter(samples, variable_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(samples)), samples, alpha=0.6)
    plt.title(f"1D Scatter Plot for {variable_name}")
    plt.xlabel("Sample Index")
    plt.ylabel(variable_name)
    plt.grid()
    plt.show()

def plot_2d_scatter(x, y, x_name, y_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.6)
    plt.title(f"2D Scatter Plot: {x_name} vs {y_name}")
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.grid()
    plt.show()

class ManualInputMode(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manual Input Mode")
        self.setGeometry(100, 100, 1200, 800)

        self.input_table = QTableWidget(0, 5)
        self.input_table.setHorizontalHeaderLabels(["Name", "Type", "Default", "Min", "Max"])

        self.formula_table = QTableWidget(0, 2)
        self.formula_table.setHorizontalHeaderLabels(["Output Name", "Formula"])

        self.sampling_method_dropdown = QComboBox()
        self.sampling_method_dropdown.addItems([
            "Monte Carlo", "Quasi Monte Carlo", "Latin Hypercube", "Orthogonal Array",
            "Gradient Sample", "METIS", "Sobol", "Morris Design", "Generalized Morris Design"
        ])

        self.samples_input = QLineEdit()
        self.samples_input.setPlaceholderText("Number of Samples")

        self.run_sampling_btn = QPushButton("Generate Samples")
        self.run_sampling_btn.clicked.connect(self.generate_samples)

        self.add_input_btn = QPushButton("Add Input Variable")
        self.add_input_btn.clicked.connect(self.add_input_variable)

        self.add_formula_btn = QPushButton("Add Formula")
        self.add_formula_btn.clicked.connect(self.add_formula)

        self.samples_table = QTableWidget()

        # Visualization Buttons
        self.select_variable_list = QListWidget()
        self.select_variable_list.setSelectionMode(QListWidget.MultiSelection)
        self.update_variable_list_btn = QPushButton("Update Variable List")
        self.update_variable_list_btn.clicked.connect(self.update_variable_list)
        self.plot_selected_btn = QPushButton("Plot Selected Variables")
        self.plot_selected_btn.clicked.connect(self.plot_selected_variables)
        self.analyze_results_btn = QPushButton("Analyze Results")
        self.analyze_results_btn.clicked.connect(self.analyze_results)

        # Allowed Functions Label
        self.allowed_functions_label = QLabel(
            "Allowed functions: sin, cos, tan, exp, log, sqrt, abs, ceil, floor"
        )
        self.allowed_functions_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Input Variables:"))
        layout.addWidget(self.input_table)
        layout.addWidget(self.add_input_btn)

        layout.addWidget(QLabel("Formulas:"))
        layout.addWidget(self.formula_table)
        layout.addWidget(self.add_formula_btn)
        layout.addWidget(self.allowed_functions_label)

        sampling_layout = QHBoxLayout()
        sampling_layout.addWidget(QLabel("Sampling Method:"))
        sampling_layout.addWidget(self.sampling_method_dropdown)
        sampling_layout.addWidget(QLabel("Number of Samples:"))
        sampling_layout.addWidget(self.samples_input)
        sampling_layout.addWidget(self.run_sampling_btn)
        layout.addLayout(sampling_layout)

        layout.addWidget(QLabel("Generated Samples and Outputs:"))
        layout.addWidget(self.samples_table)

        layout.addWidget(QLabel("Select Variables for Plotting:"))
        layout.addWidget(self.select_variable_list)
        layout.addWidget(self.update_variable_list_btn)
        layout.addWidget(self.plot_selected_btn)
        layout.addWidget(self.analyze_results_btn)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.samples = None
        self.outputs = {}
        self.variable_names = []

    def add_input_variable(self):
        row = self.input_table.rowCount()
        self.input_table.insertRow(row)
        self.input_table.setItem(row, 0, QTableWidgetItem(f"Var{row + 1}"))
        self.input_table.setItem(row, 1, QTableWidgetItem("float"))
        self.input_table.setItem(row, 2, QTableWidgetItem("0"))
        self.input_table.setItem(row, 3, QTableWidgetItem("-10"))
        self.input_table.setItem(row, 4, QTableWidgetItem("10"))

    def add_formula(self):
        row = self.formula_table.rowCount()
        self.formula_table.insertRow(row)
        self.formula_table.setItem(row, 0, QTableWidgetItem(f"Output{row + 1}"))
        self.formula_table.setItem(row, 1, QTableWidgetItem(""))

    def validate_formula(self, formula):
        for var in self.variable_names:
            formula = formula.replace(var, "")
        allowed_functions = ["sin", "cos", "tan", "exp", "log", "sqrt", "abs", "ceil", "floor", "(", ")", "+", "-", "*", "/"]
        for func in allowed_functions:
            formula = formula.replace(func, "")
        return all(c.isdigit() or c.isspace() for c in formula)

    def generate_samples(self):
        try:
            bounds = []
            self.variable_names = []
            for row in range(self.input_table.rowCount()):
                name = self.input_table.item(row, 0).text()
                self.variable_names.append(name)
                min_val = float(self.input_table.item(row, 3).text())
                max_val = float(self.input_table.item(row, 4).text())
                bounds.append([min_val, max_val])
            method = self.sampling_method_dropdown.currentText()
            n_samples = int(self.samples_input.text())
            if method == "Monte Carlo":
                self.samples = monte_carlo_sample(bounds, n_samples)
            elif method == "Quasi Monte Carlo":
                self.samples = quasi_monte_carlo_sample(bounds, n_samples)
            elif method == "Latin Hypercube":
                self.samples = latin_hypercube_sample(bounds, n_samples)
            elif method == "Orthogonal Array":
                self.samples = orthogonal_array_sample(bounds, n_samples)
            elif method == "Gradient Sample":
                self.samples = gradient_sample(bounds, n_samples)
            elif method == "METIS":
                self.samples = metis_sample(bounds, n_samples)
            elif method == "Sobol":
                self.samples, _ = sobol_sample(bounds, n_samples)
            elif method == "Morris Design":
                self.samples = morris_design(bounds, n_samples)
            elif method == "Generalized Morris Design":
                self.samples = generalized_morris_design(bounds, n_samples)

            self.outputs = {}
            for row in range(self.formula_table.rowCount()):
                output_name = self.formula_table.item(row, 0).text()
                formula = self.formula_table.item(row, 1).text()
                if not self.validate_formula(formula):
                    raise ValueError(f"Invalid formula: {formula}")
                var_dict = {self.variable_names[j]: self.samples[:, j] for j in range(len(self.variable_names))}
                self.outputs[output_name] = ne.evaluate(formula, local_dict=var_dict)
            self.display_samples()
            self.update_variable_list()
            QMessageBox.information(self, "Success", "Samples and outputs generated successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating samples: {str(e)}")

    def display_samples(self):
        if self.samples is None or not self.outputs:
            return
        n_samples, n_vars = self.samples.shape
        output_names = list(self.outputs.keys())
        self.samples_table.setRowCount(n_samples)
        self.samples_table.setColumnCount(n_vars + len(output_names))
        self.samples_table.setHorizontalHeaderLabels(self.variable_names + output_names)
        for i in range(n_samples):
            for j in range(n_vars):
                self.samples_table.setItem(i, j, QTableWidgetItem(f"{self.samples[i, j]:.4f}"))
            for k, output_name in enumerate(output_names):
                self.samples_table.setItem(i, n_vars + k, QTableWidgetItem(f"{self.outputs[output_name][i]:.4f}"))

    def update_variable_list(self):
        self.select_variable_list.clear()
        self.select_variable_list.addItems(self.variable_names + list(self.outputs.keys()))

    def plot_selected_variables(self):
        selected_items = self.select_variable_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "No variables selected for plotting!")
            return

        if len(selected_items) == 1:
            var_name = selected_items[0].text()
            if var_name in self.variable_names:
                var_index = self.variable_names.index(var_name)
                data = self.samples[:, var_index]
                plot_1d_scatter(data, var_name)
            elif var_name in self.outputs:
                data = self.outputs[var_name]
                plot_1d_scatter(data, var_name)
        elif len(selected_items) == 2:
            var1, var2 = selected_items
            if var1.text() in self.variable_names:
                data1 = self.samples[:, self.variable_names.index(var1.text())]
            else:
                data1 = self.outputs[var1.text()]

            if var2.text() in self.variable_names:
                data2 = self.samples[:, self.variable_names.index(var2.text())]
            else:
                data2 = self.outputs[var2.text()]
            
            plot_2d_scatter(data1, data2, var1.text(), var2.text())
        else:
            QMessageBox.warning(self, "Warning", "Select one or two variables only for plotting!")

    def analyze_results(self):
        if not self.outputs:
            QMessageBox.warning(self, "Warning", "No outputs to analyze!")
            return

        for output_name, output_data in self.outputs.items():
            plot_output_distribution(output_data, output_name)
        QMessageBox.information(self, "Analysis", "Analysis complete. Distribution plots displayed!")

# Aspen UQ Tool
class AspenUQTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aspen UQ Tool with Multiple Sampling Methods")
        self.setGeometry(100, 100, 1200, 800)
        self.json_data = None
        self.aspen_sim = None
        self.model_file = None
        self.samples = None
        self.outputs = []

        self.input_table = QTableWidget(0, 6)
        self.input_table.setHorizontalHeaderLabels(["Sheet", "Variable", "Value", "Row", "Col", "Min", "Max"])

        self.output_table = QTableWidget(0, 5)
        self.output_table.setHorizontalHeaderLabels(["Sheet", "Variable", "Value", "Row", "Col"])

        # Generated Samples Table
        self.samples_table = QTableWidget(0, 0)
        self.samples_table.setHorizontalHeaderLabels([])

        # Results Table for Outputs
        self.results_table = QTableWidget(0, 0)
        self.results_table.setHorizontalHeaderLabels([])

        # Sampling Method Dropdown
        self.sampling_method_dropdown = QComboBox()
        self.sampling_method_dropdown.addItems([
            "Monte Carlo", "Quasi Monte Carlo", "Latin Hypercube", "Orthogonal Array",
            "Gradient Sample", "METIS", "Sobol", "Morris Design", "Generalized Morris Design"
        ])

        # Number of Samples Input
        self.num_samples_input = QLineEdit()
        self.num_samples_input.setPlaceholderText("Enter number of samples")

        # Buttons
        self.load_json_btn = QPushButton("Load JSON File")
        self.load_json_btn.clicked.connect(self.load_json)

        self.select_bkp_btn = QPushButton("Select Aspen .bkp File")
        self.select_bkp_btn.clicked.connect(self.select_bkp_file)

        self.generate_samples_btn = QPushButton("Generate Samples")
        self.generate_samples_btn.clicked.connect(self.generate_samples)

        self.run_aspen_btn = QPushButton("Run Aspen Simulations")
        self.run_aspen_btn.clicked.connect(self.run_aspen_simulations)

        # Layouts
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Input Variables:"))
        layout.addWidget(self.input_table)
        layout.addWidget(self.load_json_btn)
        layout.addWidget(self.select_bkp_btn)
        layout.addWidget(QLabel("Sampling Method:"))
        layout.addWidget(self.sampling_method_dropdown)
        layout.addWidget(QLabel("Number of Samples:"))
        layout.addWidget(self.num_samples_input)
        layout.addWidget(self.generate_samples_btn)
        layout.addWidget(QLabel("Generated Samples:"))
        layout.addWidget(self.samples_table)
        layout.addWidget(self.run_aspen_btn)
        layout.addWidget(QLabel("Output Variables:"))
        layout.addWidget(self.output_table)
        layout.addWidget(QLabel("Simulation Results for Generated Samples:"))
        layout.addWidget(self.results_table)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_json(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select JSON File", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    self.json_data = json.load(file)
                self.populate_input_table()
                self.populate_output_table()
                QMessageBox.information(self, "Success", "JSON file loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load JSON file: {e}")

    def select_bkp_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Aspen .bkp File", "", "Aspen Files (*.bkp);;All Files (*)", options=options)
        if file_path:
            self.model_file = file_path
            QMessageBox.information(self, "Success", f"Selected Aspen File: {self.model_file}")

    def populate_input_table(self):
        if not self.json_data or "inputs" not in self.json_data:
            return

        self.input_table.setRowCount(0)
        for name, details in self.json_data["inputs"].items():
            row_position = self.input_table.rowCount()
            self.input_table.insertRow(row_position)
            self.input_table.setItem(row_position, 0, QTableWidgetItem(name))
            self.input_table.setItem(row_position, 1, QTableWidgetItem(details["path"][0]))
            self.input_table.setItem(row_position, 2, QTableWidgetItem(str(details["default"])))
            self.input_table.setItem(row_position, 3, QTableWidgetItem(str(details["default"] * 0.9)))
            self.input_table.setItem(row_position, 4, QTableWidgetItem(str(details["default"] * 1.1)))

    def populate_output_table(self):
        if not self.json_data or "outputs" not in self.json_data:
            return

        self.output_table.setRowCount(0)
        for name, details in self.json_data["outputs"].items():
            row_position = self.output_table.rowCount()
            self.output_table.insertRow(row_position)
            self.output_table.setItem(row_position, 0, QTableWidgetItem(name))
            self.output_table.setItem(row_position, 1, QTableWidgetItem(details["path"][0]))
            self.output_table.setItem(row_position, 2, QTableWidgetItem(str(details["default"])))

    def generate_samples(self):
        try:
            num_samples = int(self.num_samples_input.text())
            bounds = []
            for row in range(self.input_table.rowCount()):
                min_val = float(self.input_table.item(row, 3).text())
                max_val = float(self.input_table.item(row, 4).text())
                bounds.append([min_val, max_val])

            method = self.sampling_method_dropdown.currentText()
            if method == "Monte Carlo":
                self.samples = monte_carlo_sample(bounds, num_samples)
            elif method == "Quasi Monte Carlo":
                self.samples = quasi_monte_carlo_sample(bounds, num_samples)
            elif method == "Latin Hypercube":
                self.samples = latin_hypercube_sample(bounds, num_samples)
            elif method == "Orthogonal Array":
                self.samples = orthogonal_array_sample(bounds, num_samples)
            elif method == "Gradient Sample":
                self.samples = gradient_sample(bounds, num_samples)
            elif method == "METIS":
                self.samples = metis_sample(bounds, num_samples)
            elif method == "Sobol":
                self.samples, _ = sobol_sample(bounds, num_samples)
            elif method == "Morris Design":
                self.samples = morris_design(bounds, num_samples)
            elif method == "Generalized Morris Design":
                self.samples = generalized_morris_design(bounds, num_samples)
            self.display_samples()
            QMessageBox.information(self, "Success", f"Generated {num_samples} samples using {method}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating samples: {e}")

    def display_samples(self):
        if self.samples is None:
            return

        num_samples, num_vars = self.samples.shape
        self.samples_table.setRowCount(num_samples)
        self.samples_table.setColumnCount(num_vars)
        self.samples_table.setHorizontalHeaderLabels([self.input_table.item(i, 0).text() for i in range(num_vars)])
        for i in range(num_samples):
            for j in range(num_vars):
                self.samples_table.setItem(i, j, QTableWidgetItem(str(round(self.samples[i, j], 4))))

    def run_aspen_simulations(self):
        try:
            if not self.model_file or self.samples is None:
                QMessageBox.warning(self, "Warning", "Select a .bkp file and generate samples first!")
                return
            self.aspen_sim = win32com.client.Dispatch("Apwn.Document")
            self.aspen_sim.InitFromArchive2(self.model_file)
            num_samples, num_vars = self.samples.shape
            self.outputs = []
            for i in range(num_samples):
                for j, row in enumerate(range(self.input_table.rowCount())):
                    path = self.input_table.item(row, 1).text()
                    value = self.samples[i, j]
                    self.aspen_sim.Tree.FindNode(path).Value = value

                self.aspen_sim.Engine.Run2()
                output_values = [self.aspen_sim.Tree.FindNode(self.output_table.item(row, 1).text()).Value for row in range(self.output_table.rowCount())]
                self.outputs.append(output_values)

            self.display_simulation_results()
            QMessageBox.information(self, "Success", "Aspen simulations completed successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error running simulations: {e}")

    def display_simulation_results(self):
        if not self.outputs:
            return
        num_samples = len(self.outputs)
        num_outputs = len(self.outputs[0])
        self.results_table.setRowCount(num_samples)
        self.results_table.setColumnCount(num_outputs)
        self.results_table.setHorizontalHeaderLabels([self.output_table.item(i, 0).text() for i in range(num_outputs)])
        for i, output_row in enumerate(self.outputs):
            for j, value in enumerate(output_row):
                self.results_table.setItem(i, j, QTableWidgetItem(str(round(value, 4))))

class ExcelSelectionTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Excel Selection Tool")
        self.setGeometry(100, 100, 1300, 900)
        self.excel_path = None
        self.selected_inputs = []  
        self.selected_outputs = []

        # UI Components
        self.upload_button = QPushButton("Upload Excel File")
        self.upload_button.clicked.connect(self.upload_excel)

        self.open_excel_button = QPushButton("Open Excel for Selection")
        self.open_excel_button.clicked.connect(self.open_excel)
        self.open_excel_button.setEnabled(False)

        self.get_inputs_button = QPushButton("Get Selected Inputs")
        self.get_inputs_button.clicked.connect(self.get_selected_inputs)
        self.get_inputs_button.setEnabled(False)

        self.get_outputs_button = QPushButton("Get Selected Outputs")
        self.get_outputs_button.clicked.connect(self.get_selected_outputs)
        self.get_outputs_button.setEnabled(False)

        self.run_monte_carlo_btn = QPushButton("Run Monte Carlo Simulation")
        self.run_monte_carlo_btn.clicked.connect(self.run_monte_carlo)
        self.run_monte_carlo_btn.setEnabled(True)

        self.input_table = QTableWidget(0, 5)  
        self.input_table.setHorizontalHeaderLabels(["Sheet", "Input Name","Value", "Min", "Max"])

        self.output_table = QTableWidget(0, 3)
        self.output_table.setHorizontalHeaderLabels(["Sheet", "Output Name", "Value"])

        # Sampling parameters UI
        self.num_samples_input = QLineEdit()
        self.num_samples_input.setPlaceholderText("Enter Number of Samples")

        self.min_factor_input = QLineEdit()
        self.min_factor_input.setPlaceholderText("Min Factor (default: 0.5)")

        self.max_factor_input = QLineEdit()
        self.max_factor_input.setPlaceholderText("Max Factor (default: 1.5)")

        self.std_dev_factor_input = QLineEdit()
        self.std_dev_factor_input.setPlaceholderText("Std Dev Factor (default: 1/6)")

        # Tables for Monte Carlo Samples
        self.samples_table = QTableWidget(0, 0)
        self.samples_table.setHorizontalHeaderLabels([])

        self.output_variation_table = QTableWidget(0, 0)
        self.output_variation_table.setHorizontalHeaderLabels([])

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Upload an Excel file for processing:"))
        layout.addWidget(self.upload_button)
        layout.addWidget(self.open_excel_button)
        layout.addWidget(self.get_inputs_button)
        layout.addWidget(QLabel("Selected Input Variables:"))
        layout.addWidget(self.input_table)
        layout.addWidget(self.get_outputs_button)
        layout.addWidget(QLabel("Selected Output Variables:"))
        layout.addWidget(self.output_table)

        # Monte Carlo parameter input layout
        mc_layout = QHBoxLayout()
        mc_layout.addWidget(QLabel("Num Samples:"))
        mc_layout.addWidget(self.num_samples_input)
        mc_layout.addWidget(QLabel("Min Factor:"))
        mc_layout.addWidget(self.min_factor_input)
        mc_layout.addWidget(QLabel("Max Factor:"))
        mc_layout.addWidget(self.max_factor_input)
        mc_layout.addWidget(QLabel("Std Dev Factor:"))
        mc_layout.addWidget(self.std_dev_factor_input)
        layout.addLayout(mc_layout)
        layout.addWidget(self.run_monte_carlo_btn)
        layout.addWidget(QLabel("Generated Monte Carlo Input Samples:"))
        layout.addWidget(self.samples_table)
        layout.addWidget(QLabel("Monte Carlo Output Variation:"))
        layout.addWidget(self.output_variation_table)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def upload_excel(self):
        start_time = time.perf_counter()
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Excel File", "", "Excel Files (*.xlsx *.xls);;All Files (*)", options=options)
        if file_path:
            self.excel_path = file_path
            self.open_excel_button.setEnabled(True)
            elapsed_time = time.perf_counter() - start_time
            print(f"[TIMER] Excel file uploaded in {elapsed_time:.4f} seconds.")
            QMessageBox.information(self, "Success", "Excel file uploaded successfully!")
 
    def open_excel(self):
        if self.excel_path:
            start_time = time.perf_counter()
            os.startfile(self.excel_path)
            self.get_inputs_button.setEnabled(True)
            self.get_outputs_button.setEnabled(True)
            elapsed_time = time.perf_counter() - start_time
            print(f"[TIMER] Excel file opened in {elapsed_time:.4f} seconds.")
            QMessageBox.information(self, "Info", "Please select variables in Excel, then click 'Get Selected Inputs/Outputs'.")

    def get_selected_inputs(self):
        try:
            #start_time = time.perf_counter()
            print("\n[DEBUG] Fetching selected input values from Excel...")

            # Connect to Excel
            self.excel_app = win32com.client.Dispatch("Excel.Application")
            self.excel_workbook = self.excel_app.Workbooks.Open(self.excel_path)

            active_sheet = self.excel_workbook.ActiveSheet
            selected_range = active_sheet.Application.Selection
            sheet_name = active_sheet.Name  # Capture sheet name
            start_time = time.perf_counter()
            if selected_range is None or selected_range.Cells.Count == 0:
                QMessageBox.warning(self, "Warning", "No selection detected for inputs. Please select a value cell in Excel.")
                return

            selected_data = []
            print(f"[DEBUG] Active Sheet: {sheet_name}")

            for cell in selected_range:
                row, col = cell.Row, cell.Column
                value = active_sheet.Cells(row, col).Value  # Capture the selected value

                if value is None:
                    continue  # Skip empty selections

                # Capture the variable name from the left cell (same row, col - 1)
                variable_name = active_sheet.Cells(row, col - 1).Value if col > 1 else f"Var_{row}_{col}"

                # Set min and max range based on default factors
                min_val = 0.5 * value
                max_val = 1.5 * value

                # Store (Sheet Name, Variable Name, Value, Row, Column, Min, Max)
                selected_data.append((sheet_name, variable_name, value, row, col, min_val, max_val))
                print(f"[DEBUG] Captured Input: {variable_name} = {value}, Location: {sheet_name} ({row}, {col})")

            self.selected_inputs.extend(selected_data)

            # Update input table
            self.input_table.setRowCount(len(self.selected_inputs))
            for i, (sheet, var_name, val, row, col, min_val, max_val) in enumerate(self.selected_inputs):
                self.input_table.setItem(i, 0, QTableWidgetItem(sheet))
                self.input_table.setItem(i, 1, QTableWidgetItem(var_name))
                self.input_table.setItem(i, 2, QTableWidgetItem(str(val)))  # Only the value
                self.input_table.setItem(i, 3, QTableWidgetItem(str(min_val)))
                self.input_table.setItem(i, 4, QTableWidgetItem(str(max_val)))
            
            QMessageBox.information(self, "Success", "Inputs captured successfully!")
            elapsed_time = time.perf_counter() - start_time
            print(f"[TIMER] Inputs captured in {elapsed_time:.4f} seconds.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to get inputs from Excel: {e}")
            print(f"[ERROR] {e}")

    def get_selected_outputs(self):
        try:
            
            print("\n[DEBUG] Fetching selected output values from Excel...")
            # Connect to Excel
            self.excel_app = win32com.client.Dispatch("Excel.Application")
            self.excel_workbook = self.excel_app.Workbooks.Open(self.excel_path)
            active_sheet = self.excel_workbook.ActiveSheet
            selected_range = active_sheet.Application.Selection
            sheet_name = active_sheet.Name  # Capture sheet name

            start_time = time.perf_counter()
            if selected_range is None or selected_range.Cells.Count == 0:
                QMessageBox.warning(self, "Warning", "No selection detected for outputs. Please select a value cell in Excel.")
                return

            selected_data = []
            print(f"[DEBUG] Active Sheet: {sheet_name}")

            for cell in selected_range:
                row, col = cell.Row, cell.Column
                value = active_sheet.Cells(row, col).Value  # Capture the selected value

                if value is None:
                    continue  # Skip empty selections

                # Capture the variable name from the left cell (same row, col - 1)
                variable_name = active_sheet.Cells(row, col - 1).Value if col > 1 else f"Output_{row}_{col}"

                selected_data.append((sheet_name, variable_name, value, row, col))
                print(f"[DEBUG] Captured Output: {variable_name} = {value}, Location: {sheet_name} ({row}, {col})")

            self.selected_outputs.extend(selected_data)

            # Update output table
            self.output_table.setRowCount(len(self.selected_outputs))
            for i, (sheet, var_name, val, row, col) in enumerate(self.selected_outputs):
                self.output_table.setItem(i, 0, QTableWidgetItem(sheet))
                self.output_table.setItem(i, 1, QTableWidgetItem(var_name))
                self.output_table.setItem(i, 2, QTableWidgetItem(str(val)))  # Only the value

            QMessageBox.information(self, "Success", "Outputs captured successfully!")
            elapsed_time = time.perf_counter() - start_time
            print(f"[TIMER] Outputs captured in {elapsed_time:.4f} seconds.") 
 
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to get outputs from Excel: {e}")
            print(f"[ERROR] {e}")

    def generate_random_samples(self):
        try:
            print(f"\n[DEBUG] Generating random samples...")
            sample_start_time = time.perf_counter()
            if not self.selected_inputs:
                QMessageBox.warning(self, "Warning", "No input variables selected.")
                return None

            num_samples = int(self.num_samples_input.text())
            min_factor = float(self.min_factor_input.text()) if self.min_factor_input.text() else 0.5
            max_factor = float(self.max_factor_input.text()) if self.max_factor_input.text() else 1.5
            std_dev_factor = float(self.std_dev_factor_input.text()) if self.std_dev_factor_input.text() else 1/6
            random_samples = []

            for sheet, var_name, value, row, col, min_val, max_val in self.selected_inputs:
                
                mean = value
                std_dev = std_dev_factor * (max_val - min_val)
                samples = np.random.normal(mean, std_dev, num_samples)
                samples = np.clip(samples, min_val, max_val)
                #print(f"[DEBUG] Generated samples for {var_name}: {samples}")  # Debugging output
                random_samples.append((sheet, var_name, row, col, samples))  
            elapsed_sample_time = time.perf_counter() - sample_start_time
            print(f"[TIMER] Sample generation for {var_name} took {elapsed_sample_time:.4f} seconds.")
            return random_samples

        
        except Exception as e:
            print(f"[ERROR] Failed to generate random samples: {e}")
            return None

    def run_monte_carlo(self):
        try:
            print("\n[DEBUG] Starting Monte Carlo Simulation...")
            total_start = time.perf_counter()
            
            # Sample Generation
            sample_gen_start = time.perf_counter()
            generated_samples = self.generate_random_samples()
            sample_gen_time = time.perf_counter() - sample_gen_start
            print(f"[TIMER] Sample generation took {sample_gen_time:.4f} seconds.")

            if not generated_samples:
                print("[ERROR] No generated samples. Stopping simulation.")
                return

            num_samples = len(generated_samples[0][4])
            output_results = []
            input_results = []

            # Excel Load
            excel_load_start = time.perf_counter()
            self.excel_app = win32com.client.Dispatch("Excel.Application")
            self.excel_workbook = self.excel_app.Workbooks.Open(self.excel_path)
            self.excel_app.ScreenUpdating = False
            self.excel_app.EnableEvents = False
            excel_load_time = time.perf_counter() - excel_load_start
            print(f"[TIMER] Excel loading/setup took {excel_load_time:.4f} seconds.")         
            
            # Original Value Caching
            cache_start = time.perf_counter()
            self.original_input_values = {
                (sheet, row, col): self.excel_workbook.Sheets(sheet).Cells(row, col).Value
                for sheet, var_name, row, col, _ in generated_samples
            }
            cache_time = time.perf_counter() - cache_start
            print(f"[TIMER] Original value caching took {cache_time:.4f} seconds.")

            # Run Simulations
            sim_start = time.perf_counter()
            for i in range(num_samples):
                iter_start = time.perf_counter()
                current_input_values = []

                write_start = time.perf_counter()
                for sheet, var_name, row, col, samples in generated_samples:
                    value_to_write = samples[i]
                    active_sheet = self.excel_workbook.Sheets(sheet)
                    active_sheet.Cells(row, col).Value = value_to_write
                    current_input_values.append(samples[i])
                    print(f"[DEBUG] Sample {i+1}: Writing {value_to_write} to '{var_name}' at {sheet}!R{row}C{col}")
                input_results.append(current_input_values)
                
                write_time = time.perf_counter() - write_start
                print(f"[TIMER] Sample {i+1}: input write time = {write_time:.4f}s")
                
                read_output_start = time.perf_counter()
                output_values = []
                for output in self.selected_outputs:
                    sheet, var_name, _, row, col = output
                    active_sheet = self.excel_workbook.Sheets(sheet)
                    output_value = active_sheet.Cells(row, col).Value
                    output_values.append(output_value)
                output_results.append(output_values)
                read_output_time = time.perf_counter() - read_output_start

                iter_time = time.perf_counter() - iter_start
                print(f"[TIMER] Sample {i+1}/{num_samples}: write={write_time:.4f}s, read_output={read_output_time:.4f}s, total_iter={iter_time:.4f}s")

            sim_time = time.perf_counter() - sim_start
            print(f"[TIMER] Monte Carlo simulation iterations took {sim_time:.4f} seconds.")

            # Restore original values
            restore_start = time.perf_counter()
            for (sheet, row, col), original_value in self.original_input_values.items():
                self.excel_workbook.Sheets(sheet).Cells(row, col).Value = original_value
            restore_time = time.perf_counter() - restore_start
            print(f"[TIMER] Restoring original Excel values took {restore_time:.4f} seconds.")

            # Convert results to DataFrame
            df_start = time.perf_counter()
            df_inputs = pd.DataFrame(input_results, columns=[var[1] for var in self.selected_inputs])
            df_outputs = pd.DataFrame(output_results, columns=[var[1] for var in self.selected_outputs])
            df_time = time.perf_counter() - df_start
            print(f"[TIMER] DataFrame conversion took {df_time:.4f} seconds.")

            # Restore Excel settings
            self.excel_app.ScreenUpdating = True
            self.excel_app.EnableEvents = True

            # GUI Update
            gui_start = time.perf_counter()

            # Inputs Table
            gui_inputs_start = time.perf_counter()
            self.samples_table.setRowCount(df_inputs.shape[0])
            self.samples_table.setColumnCount(df_inputs.shape[1])
            self.samples_table.setHorizontalHeaderLabels(df_inputs.columns)
            for i in range(df_inputs.shape[0]):
                for j in range(df_inputs.shape[1]):
                    self.samples_table.setItem(i, j, QTableWidgetItem(str(df_inputs.iloc[i, j])))
            gui_inputs_time = time.perf_counter() - gui_inputs_start

            # Outputs Table
            gui_outputs_start = time.perf_counter()
            self.output_variation_table.setRowCount(df_outputs.shape[0])
            self.output_variation_table.setColumnCount(df_outputs.shape[1])
            self.output_variation_table.setHorizontalHeaderLabels(df_outputs.columns)
            for i in range(df_outputs.shape[0]):
                for j in range(df_outputs.shape[1]):
                    self.output_variation_table.setItem(i, j, QTableWidgetItem(str(df_outputs.iloc[i, j])))
            gui_outputs_time = time.perf_counter() - gui_outputs_start

            gui_time = time.perf_counter() - gui_start
            print(f"[TIMER] GUI update: inputs={gui_inputs_time:.4f}s, outputs={gui_outputs_time:.4f}s, total={gui_time:.4f}s")

            # Visualization
            viz_start = time.perf_counter()
            self.generate_scatter_plots(df_inputs, df_outputs)
            self.generate_visualizations(df_inputs, df_outputs)
            viz_time = time.perf_counter() - viz_start
            print(f"[TIMER] Visualizations took {viz_time:.4f} seconds.")

            total_elapsed = time.perf_counter() - total_start
            print(f"[TIMER] Total Monte Carlo simulation completed in {total_elapsed:.4f} seconds.")
            QMessageBox.information(self, "Success", "Monte Carlo Simulation completed!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Monte Carlo Simulation failed: {e}")
            print(f"[ERROR] {e}")


    def generate_scatter_plots(self, df_inputs, df_outputs):
        if df_inputs.empty or df_outputs.empty:
            print("[ERROR] No data to plot.")
            return
        fig, axes = plt.subplots(1, len(df_inputs.columns), figsize=(10 * len(df_inputs.columns), 5))
        if len(df_inputs.columns) == 1:
            axes = [axes]  # Ensure iterable for single-input case
        colors = ['orange', 'red']  # First input → orange, second input → red
        for idx, (input_col, input_name) in enumerate(zip(df_inputs.columns, [var[1] for var in self.selected_inputs])):
            color = colors[idx % len(colors)]  # Assign colors in sequence
            axes[idx].scatter(
                df_inputs[input_col], df_outputs.iloc[:, 0],
                color=color, marker='x', alpha=0.8, s=25  # Use 'x' marker for cross effect
            )
            axes[idx].set_xlabel(input_name, fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(self.selected_outputs[0][1], fontsize=12, fontweight='bold')
            axes[idx].set_title(f"{input_name} vs. {self.selected_outputs[0][1]}", fontsize=14, fontweight='bold')

            # Add grid and background
            axes[idx].grid(True, linestyle='--', linewidth=0.5)
            axes[idx].set_facecolor('whitesmoke')
        plt.tight_layout()
        plt.show()

    def generate_visualizations(self, df_inputs, df_outputs):
        if df_inputs.empty or df_outputs.empty:
            print("[ERROR] No data to plot.")
            return
        num_inputs = len(df_inputs.columns)
        fig, axes = plt.subplots(2, num_inputs, figsize=(12 * num_inputs, 10))

        # Ensure axes remain iterable (handles single input case)
        if num_inputs == 1:
            axes = np.array(axes).reshape(2, 1)  # Convert to 2D array

        colors = ['orange', 'red']  # Colors for scatter plots

        for idx, input_col in enumerate(df_inputs.columns):
            color = colors[idx % len(colors)]  # Assign colors in sequence
            axes[0, idx].scatter(
                df_inputs[input_col], df_outputs.iloc[:, 0],
                color=color, marker='x', alpha=0.7, s=20
            )
            axes[0, idx].set_xlabel(input_col, fontsize=12, fontweight='bold')
            axes[0, idx].set_ylabel(df_outputs.columns[0], fontsize=12, fontweight='bold')
            axes[0, idx].set_title(f"{input_col} vs. {df_outputs.columns[0]}", fontsize=14, fontweight='bold')
            axes[0, idx].grid(True, linestyle='--', linewidth=0.5)
            axes[0, idx].set_facecolor('whitesmoke')

            output_values = df_outputs.iloc[:, 0].values
            bins = np.linspace(min(output_values), max(output_values), 50)  # More bins for thinner bars

            counts, bin_edges, _ = axes[1, idx].hist(
                output_values, bins=bins, density=False, alpha=0.7, 
                color='deepskyblue', edgecolor='black', linewidth=0.7, label="Frequency"
            )

            axes[1, idx].set_xlabel(f"{self.selected_outputs[0][1]}", fontsize=12, fontweight='bold')
            axes[1, idx].set_ylabel("Frequency", fontsize=12, fontweight='bold', color='deepskyblue')
            axes[1, idx].set_title(f"Probability  & Cumulative Frequency of {df_outputs.columns[0]}", fontsize=14, fontweight='bold')
            axes[1, idx].grid(True, linestyle='--', linewidth=0.5)

            # Cumulative Frequency
            cumulative_counts = np.cumsum(counts)
            cumulative_percentage = (cumulative_counts / cumulative_counts[-1]) * 100  # Convert to percentage
            ax_cdf = axes[1, idx].twinx()  # Secondary Y-axis for CDF
            ax_cdf.plot(bin_edges[1:], cumulative_percentage, color='orange', linestyle='-', linewidth=2, label="Cumulative %")
            ax_cdf.set_ylabel("Cumulative Percentage", fontsize=12, fontweight='bold', color='orange')
        plt.tight_layout()
        plt.show()

class MainUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Uncertainty Quantification Tool")
        self.setGeometry(100, 100, 800, 500)
        self.setStyleSheet("background-color: #F4F4F4;")  # Light gray background

        self.init_ui()

    def init_ui(self):
        # Title Label
        title_label = QLabel("Uncertainty Quantification Tool")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 28px; 
            font-weight: bold; 
            color: #003366;
            margin-bottom: 10px;
        """)

        # Subtitle Label
        subtitle_label = QLabel("Select an Option Below:")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 18px; color: #555; margin-bottom: 20px;")

        # Buttons with Modern Styling
        self.manual_input_btn = QPushButton("Manual Input")
        self.aspen_UQ_btn = QPushButton("Aspen UQ Tool")
        self.excel_UQ_btn = QPushButton("Excel Integration Tool")

        # Button Styling
        button_style = """
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                background-color: #003366;
                color: white;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #005599;
            }
        """
        self.manual_input_btn.setStyleSheet(button_style)
        self.aspen_UQ_btn.setStyleSheet(button_style)
        self.excel_UQ_btn.setStyleSheet(button_style)

        # Button Connections
        self.manual_input_btn.clicked.connect(self.open_manual_input)
        self.aspen_UQ_btn.clicked.connect(self.open_aspen_tool)
        self.excel_UQ_btn.clicked.connect(self.open_excel_tool)

        # Group Box for Buttons
        button_box = QGroupBox("Select a Mode")
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.manual_input_btn)
        button_layout.addWidget(self.aspen_UQ_btn)
        button_layout.addWidget(self.excel_UQ_btn)
        button_box.setLayout(button_layout)
        button_box.setStyleSheet("font-size: 16px; color: #003366; padding: 10px; border: 2px solid #003366; border-radius: 10px;")

        # Main Layout
        layout = QVBoxLayout()
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addWidget(button_box)
        layout.setAlignment(Qt.AlignCenter)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_manual_input(self):
        self.manual_input_window = ManualInputMode()
        self.manual_input_window.show()

    def open_aspen_tool(self):
        self.aspen_tool_window = AspenUQTool()
        self.aspen_tool_window.show()

    def open_excel_tool(self):
        self.excel_tool_window = ExcelSelectionTool()
        self.excel_tool_window.show()

# Run Application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainUI()
    main_window.show()
    sys.exit(app.exec_())
