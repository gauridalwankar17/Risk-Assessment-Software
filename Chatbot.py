# ==============================================
#  Uncertainty‑Quantification Tool + Chatbot UI
#  ------------------------------------------------
#  Single‑file Python script (PyQt5)
#  ------------------------------------------------
#  ▸ Run          :  python uq_tool_chatbot.py
#  ▸ Requirements :  PyQt5, numpy, numexpr, pyDOE, SALib,
#                    scikit‑learn, matplotlib, tensorflow (optional)
#                    win32com (for Excel / Aspen features, Windows‑only)
#  ▸ Purpose      :  Adds a conversational side‑panel (chatbot)
#                    that can answer questions about sampling
#                    methods, expected inputs, interpretation
#                    of plots, etc., while letting the main GUI
#                    work as before.
#
#  NOTE
#  ----
#  ▸ The chatbot logic is *intent‑based* and loads a tiny
#    pre‑trained model (`chatbot_model.keras`). If you don’t
#    have that model yet, run **train_chatbot.ipynb** (see
#    repo) once—it will generate both the model and the
#    `intentsbot.json` file.
#  ▸ If TensorFlow isn’t available, switch `USE_TF_MODEL` to
#    False and the bot will fall back to a rule‑based FAQ.
# ==============================================
from __future__ import annotations

import os, sys, json, random, time, math
from pathlib import Path
from typing import List

import numpy as np
import numexpr as ne
from pyDOE import lhs
from SALib.sample import saltelli
from sklearn.cluster import KMeans

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QLineEdit, QTextEdit, QSplitter, QMessageBox, QListWidget,
    QTableWidget, QTableWidgetItem, QComboBox, QFileDialog, QGroupBox,
)
import matplotlib.pyplot as plt

# --------------------------------------------------
#  Chatbot model (intent classifier + responses)
# --------------------------------------------------
USE_TF_MODEL = True  # change to False if TF unavailable
MODEL_PATH   = "chatbot_model.keras"
INTENTS_PATH = "intentsbot.json"

if USE_TF_MODEL:
    try:
        from tensorflow.keras.models import load_model
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize
        import string, nltk
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)
        lemmatizer = WordNetLemmatizer()

        model      = load_model(MODEL_PATH)
        with open(INTENTS_PATH) as f:
            intents_data = json.load(f)

        words   = intents_data["metadata"]["words"]     # saved during training
        classes = intents_data["metadata"]["classes"]

        def _clean(sentence:str):
            tokens = word_tokenize(sentence)
            tokens = [lemmatizer.lemmatize(t.lower()) for t in tokens if t not in string.punctuation]
            return tokens

        def _bow(sentence:str)->np.ndarray:
            sent_words = _clean(sentence)
            bag = [0]*len(words)
            for s in sent_words:
                for i,w in enumerate(words):
                    if w==s:
                        bag[i]=1
            return np.array(bag)

        def classify(sentence:str):
            probs = model.predict(np.array([_bow(sentence)]), verbose=0)[0]
            threshold=0.15
            results=[[i,p] for i,p in enumerate(probs) if p>threshold]
            results.sort(key=lambda x: x[1], reverse=True)
            return [{"intent":classes[i], "prob":float(p)} for i,p in results]

        def get_response(sentence:str)->str:
            intents = classify(sentence)
            if not intents:
                return "Sorry, I’m not sure I understand."
            top = intents[0]["intent"]
            for it in intents_data["intents"]:
                if it["tag"]==top:
                    return random.choice(it["responses"])
    except Exception as e:
        print("[Chatbot] TensorFlow path failed →", e)
        USE_TF_MODEL=False

if not USE_TF_MODEL:
    # -----------------------------
    #  Very small rule‑based fallback
    # -----------------------------
    FAQ = {
        "sampling": "The tool supports Monte‑Carlo, Sobol, Latin‑Hypercube and more. Choose a method, set sample size and click ‘Generate Samples’.",
        "monte carlo": "Monte‑Carlo draws random samples from uniform or normal distributions to explore uncertainty space.",
        "sobol": "Sobol sequences are quasi‑random low‑discrepancy samples that fill the space more evenly than pure random sampling.",
        "plot": "After generating samples, select variables in the list and press ‘Plot Selected Variables’ to visualize.",
        "excel": "Use the Excel Integration Tool button, upload your workbook, select cells as inputs/outputs and run Monte‑Carlo.",
        "aspen": "With a .bkp file and JSON variable map you can launch Aspen simulations under the Aspen UQ Tool.",
    }
    def get_response(sentence:str)->str:
        s=sentence.lower()
        for key,ans in FAQ.items():
            if key in s:
                return ans
        return "I’m sorry, could you rephrase? I’m still learning."

# --------------------------------------------------
#  Chatbot widget (dockable)
# --------------------------------------------------
class ChatDock(QWidget):
    """A simple chat panel with history view and input line."""

    def __init__(self, parent:QWidget|None=None):
        super().__init__(parent)
        self.history = QTextEdit(readOnly=True)
        self.input   = QLineEdit()
        self.send_btn= QPushButton("Send")
        self.send_btn.clicked.connect(self._on_send)
        self.input.returnPressed.connect(self._on_send)

        layout = QVBoxLayout(); row=QHBoxLayout()
        row.addWidget(self.input); row.addWidget(self.send_btn)
        layout.addWidget(self.history); layout.addLayout(row)
        self.setLayout(layout)
        self._add_bot("Hello! Ask me anything about the UQ tool.")

    def _add_user(self, msg:str):
        self.history.append(f"<b>You:</b> {msg}")

    def _add_bot(self, msg:str):
        self.history.append(f"<b>Bot:</b> {msg}")

    def _on_send(self):
        msg=self.input.text().strip()
        if not msg: return
        self.input.clear(); self._add_user(msg)
        reply=get_response(msg)
        self._add_bot(reply)

# --------------------------------------------------
#  ▼▼▼  INSERT ***YOUR EXISTING GUI CODE*** BELOW  ▼▼▼
# --------------------------------------------------
#  To keep this file concise, we import your existing
#  MainUI (and all other sampling classes) from a second
#  module **uq_tool_core.py**. Move all the gigantic code
#  you provided into that module *unchanged* and expose a
#  `create_main_window()` function that returns MainUI.
#  ----------------------------------------------------
#  from uq_tool_core import create_main_window
# --------------------------------------------------
#  BUT — for demo purposes we’ll stub a minimal window.
# --------------------------------------------------
class DummyTool(QWidget):
    def __init__(self):
        super().__init__()
        lab = QLabel("<h2 style='color:#555'>[Your UQ GUI loads here]</h2>")
        v=QVBoxLayout(); v.addWidget(lab); self.setLayout(v)

# --------------------------------------------------
#  Combined Main Window with splitter ‑‑ left: GUI, right: Chat
# --------------------------------------------------
class UnifiedWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UQ Tool with Chat Assistant")
        self.resize(1300,800)

        # Left = existing tool, Right = chatdock
        left  = DummyTool()          # replace with create_main_window() in real use
        right = ChatDock()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0,4)
        splitter.setStretchFactor(1,1)

        self.setCentralWidget(splitter)

# --------------------------------------------------
#  Launch
# --------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = UnifiedWindow()
    win.show()
    sys.exit(app.exec_())
