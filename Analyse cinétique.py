from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QPushButton, QLineEdit, QFrame, QHBoxLayout,
    QFileDialog, QTabWidget, QSizePolicy,
    QDialog, QScrollArea
)
from PyQt5.QtCore import Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import numpy as np
import sys
from scipy.optimize import least_squares


# ============================================================

class FenetreGraphes(QWidget):
    def __init__(self, tabs: QTabWidget, on_close_callback):
        super().__init__()
        self._tabs = tabs
        self._on_close_callback = on_close_callback

        self.setWindowTitle("Graphes — Analyse Cinétique")
        self.setStyleSheet("background-color: #121212; font-family: Georgia")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs.setParent(self)
        layout.addWidget(self._tabs, 1)

        self.showMaximized()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if callable(self._on_close_callback):
            self._on_close_callback()
        event.accept()


# ============================================================
class FenetreResume(QWidget):
    def __init__(self, build_callback, on_close_callback=None):
        super().__init__()
        self._build_callback = build_callback
        self._on_close_callback = on_close_callback

        self.setWindowTitle("Résumé — toutes vos valeurs")
        self.setStyleSheet("background-color: #121212; font-family: Georgia")

        self._main = QVBoxLayout(self)
        self._main.setContentsMargins(12, 12, 12, 12)
        self._main.setSpacing(10)

        if callable(self._build_callback):
            self._build_callback(self)

        self.showMaximized()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if callable(self._on_close_callback):
            self._on_close_callback()
        event.accept()


# ============================================================
def modele_X(t, k, n, eps=1e-8):
    t = np.asarray(t, float)
    k = np.asarray(k, float)
    return np.where(
        np.abs(n - 1) < eps,
        1.0 - np.exp(-k * t),
        1.0 - np.exp(np.log1p((n - 1) * k * t) / (1.0 - n))
    )


def t_demi(n, k, eps=1e-8):
    k = np.asarray(k, float)
    return np.where(np.abs(n - 1) < eps,
                    np.log(2) / k,
                    (2 ** (n - 1) - 1) / ((n - 1) * k))


def fit_n_et_k(t, Xnorm):
    t = np.asarray(t, float)
    Xnorm = np.asarray(Xnorm, float)

    def residu(p):
        n = p[0]
        k = p[1:]
        Xm = modele_X(t[:, None], k, n)
        return (Xnorm - Xm).ravel()

    p0 = np.r_[1.0, np.full(Xnorm.shape[1], 0.1)]
    res = least_squares(residu, p0, bounds=(0.0, np.inf))
    return res.x[0], res.x[1:]


def normaliser_X(X):
    X = np.asarray(X, float)
    denom = X[-1, :].copy()
    fallback = np.max(np.abs(X), axis=0)
    denom = np.where(np.abs(denom) > 1e-12, denom, fallback)
    denom = np.where(np.abs(denom) > 1e-12, denom, 1.0)
    Xn = np.clip(X / denom, 0.0, 1.0)
    return Xn, denom


# ============================================================
def principal():
    app = QApplication(sys.argv)

    # ----- Fenêtre principale -----
    fenêtre = QWidget()
    fenêtre.setWindowTitle("Analyse Cinétique")
    fenêtre.setStyleSheet("background-color: #121212; font-family: Georgia")

    principal_layout = QVBoxLayout(fenêtre)
    principal_layout.setContentsMargins(0, 0, 0, 0)
    principal_layout.setSpacing(0)

    # ----- Arrière plan -----
    cadre = QFrame()
    cadre.setStyleSheet("""
        QFrame {
            background-color: #1E1E1E;
            border-radius: 20px;
            padding: 30px;
        }
    """)
    cadre_layout = QVBoxLayout(cadre)
    cadre_layout.setSpacing(14)

    # ----- Titre -----
    title = QLabel("Analyse Cinétique")
    title.setStyleSheet("font-size: 32px; color: white; font-weight: bold;")
    cadre_layout.addWidget(title)

    # ===========================================================
    def popup_blanc(message: str, titre="Info"):
        dialog = QDialog(fenêtre)
        dialog.setWindowTitle(titre)
        dialog.setMinimumSize(700, 450)
        dialog.setAutoFillBackground(True)

        dialog.setStyleSheet("""
            QDialog {
                background-color: white;
                border-radius: 10px;
            }
            QScrollArea {
                background-color: white;
                border: none;
            }
            QScrollArea QWidget {
                background-color: white;
            }
            QLabel {
                background-color: white;
                color: black;
                font-size: 14px;
            }
            QPushButton {
                background-color: white;
                color: black;
                border: 1px solid #CCCCCC;
                border-radius: 6px;
                padding: 6px 12px;
                min-width: 90px;
            }
            QPushButton:hover { background-color: #F2F2F2; }
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        scroll.setAutoFillBackground(True)
        scroll.viewport().setAutoFillBackground(True)

        content = QWidget()
        content.setStyleSheet("background-color: white;")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(message)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setStyleSheet("background-color: white; color: black; font-size: 14px;")
        content_layout.addWidget(label, 0, Qt.AlignTop)

        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        bouton_ok = QPushButton("OK")
        bouton_ok.clicked.connect(dialog.accept)
        layout.addWidget(bouton_ok, alignment=Qt.AlignCenter)

        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.exec_()

    # ============================================================
    def popup_arrhenius(temps, k_opt, n_star):
        T = np.asarray(temps, float)
        k = np.asarray(k_opt, float)
        k = np.clip(k, 1e-300, None)

        x = 1.0 / T
        y = np.log(k)

        a, b = np.polyfit(x, y, 1)       # y = a x + b
        Ea = -a * 8.314 / 1000.0         # kJ/mol

        dialog = QDialog(fenêtre)
        dialog.setWindowTitle("Arrhenius — Ea")
        dialog.setMinimumSize(950, 650)
        dialog.setStyleSheet("""
            QDialog { background-color: #1E1E1E; border-radius: 10px; }
            QLabel { color: white; font-size: 14px; }
            QPushButton {
                background-color: #00C896;
                border-radius: 10px;
                padding: 10px 14px;
                color: white;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #00a87d; }
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        fig, ax = plt.subplots()
        fig.patch.set_facecolor("#1E1E1E")
        ax.set_facecolor("#1E1E1E")

        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, dialog)
        toolbar.setStyleSheet("""
            QToolBar { background-color: #1E1E1E; border: none; spacing: 6px; }
            QToolButton {
                background-color: #AEBAB2;
                color: white;
                border-radius: 6px;
                padding: 4px;
            }
            QToolButton:hover { background-color: #3A3A3A; }
        """)

        layout.addWidget(toolbar, 0)
        layout.addWidget(canvas, 1)

        ax.plot(x, y, linestyle="None", marker="o", markersize=7,
                markerfacecolor="white", markeredgecolor="white")

        xfit = np.linspace(x.min() * 0.98, x.max() * 1.02, 200)
        yfit = a * xfit + b
        ax.plot(xfit, yfit, linewidth=2.4)

        ax.grid(alpha=0.2)
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.set_xlabel("1 / T (K⁻¹)", color="white")
        ax.set_ylabel("ln(k)", color="white")
        ax.set_title(
            f"Arrhenius — n = {float(n_star):.6g}   |   Ea = {Ea:.6g} kJ/mol",
            color="white", pad=12
        )

        fig.tight_layout()
        canvas.draw_idle()

        info = (
            f"n = {float(n_star):.6g}\n"
            f"Ea = {Ea:.6g} kJ/mol\n\n"
            "k   :\n" +
            "\n".join([f"{Ti:g} K : {ki:.8g}" for Ti, ki in zip(T, k)])
        )
        label = QLabel(info)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(label, 0)

        btn = QPushButton("OK")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn, alignment=Qt.AlignCenter)

        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.exec_()

    # --------- messages d'erreurs ----------
    def erreur_temperature():
        popup_blanc("Il manque vos données de températures (en Kelvin).", "Erreur")

    def erreur_fichiers():
        popup_blanc("Impossible de lire les fichiers sélectionnés.", "Erreur")

    def erreur_format():
        popup_blanc(
            "Format fichier invalide.\n\n"
            "Attendu : 1 ligne d’en-tête puis des lignes avec :\n"
            "t  X1  X2  X3  X4",
            "Erreur"
        )

    # ----- Zone températures -----
    temp_layout = QHBoxLayout()
    temp_inputs = []
    for i in range(4):
        temp_input = QLineEdit()
        temp_input.setPlaceholderText(f"T{i+1} (K)")
        temp_input.setStyleSheet("""
            QLineEdit {
                background-color: #2C2C2C;
                color: white;
                border-radius: 10px;
                padding: 18px;
                font-size: 14px;
            }
        """)
        temp_layout.addWidget(temp_input)
        temp_inputs.append(temp_input)
    cadre_layout.addLayout(temp_layout)

    def lire_temperatures():
        temps = []
        try:
            for inp in temp_inputs:
                val = inp.text().strip().replace("K", "")
                temps.append(float(val))
        except Exception:
            return None
        if len(temps) != 4:
            return None
        return np.array(temps, float)

    # ----- Barre import fichiers + infos -----
    top_files = QHBoxLayout()

    fichier_label = QLabel("Aucun fichier chargé")
    fichier_label.setStyleSheet("color: #BBBBBB; font-size: 14px;")
    top_files.addWidget(fichier_label, 1)

    bouton_ajouter = QPushButton("Ajouter un fichier (.txt)")
    bouton_ajouter.setStyleSheet("""
        QPushButton {
            background-color: #763d95;
            border-radius: 12px;
            padding: 12px;
            color: white;
            font-size: 14px;
        }
        QPushButton:hover { background-color: #62317d; }
    """)
    bouton_ajouter.setFocusPolicy(Qt.NoFocus)
    top_files.addWidget(bouton_ajouter)

    cadre_layout.addLayout(top_files)

    # ----- Onglets pour les graphes -----
    tabs = QTabWidget()
    tabs.setStyleSheet("""
        QTabWidget::pane { border: 0; }
        QTabBar::tab {
            background: #2C2C2C;
            color: #BBBBBB;
            padding: 10px 14px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            margin-right: 6px;
        }
        QTabBar::tab:selected {
            background: #3A3A3A;
            color: white;
        }
    """)
    tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    tabs_container = QWidget()
    tabs_container_layout = QVBoxLayout(tabs_container)
    tabs_container_layout.setContentsMargins(0, 0, 0, 0)
    tabs_container_layout.setSpacing(0)
    tabs_container_layout.addWidget(tabs, 1)
    cadre_layout.addWidget(tabs_container, 1)

    def redraw_current_tab(_=None):
        page = tabs.currentWidget()
        if page is not None and hasattr(page, "_canvas"):
            page._canvas.draw_idle()

    tabs.currentChanged.connect(redraw_current_tab)

    # ----- Lecture fichiers -----
    def lire_fichier(chemin: str):
        try:
            with open(chemin, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            try:
                with open(chemin, "r") as f:
                    return f.read()
            except Exception:
                return None

    # ----- (t + 4 colonnes) -----
    def parser_donnees(contenu: str):
        text = (contenu or "").strip()
        if text == "":
            return None

        lines = text.splitlines()
        if len(lines) < 2:
            return None

        lines = lines[1:]  # ignore header

        t = []
        X = [[], [], [], []]

        for line in lines:
            s = line.strip()
            if not s:
                continue
            values = s.replace(",", ".").split()
            if len(values) < 5:
                continue
            try:
                t.append(float(values[0]))
                for i in range(4):
                    X[i].append(float(values[i + 1]))
            except Exception:
                continue

        if len(t) < 3:
            return None

        t = np.array(t, dtype=float)
        X = np.column_stack([np.array(col, float) for col in X])
        return t, X

    # ----- Trendline via fit du modèle -----
    def trendline_modele(t, X):
        Xnorm, denom = normaliser_X(X)
        try:
            n_star, k_opt = fit_n_et_k(t, Xnorm)
            Xm_norm = modele_X(t[:, None], k_opt, n_star)
            Xm = Xm_norm * denom
            return Xm
        except Exception:
            return None

    # ----- Utilitaire : mettre toolbar en bas (à l'appui sur valider) -----
    def mettre_toolbar_en_bas(page):
        lay = getattr(page, "_layout", None)
        toolbar = getattr(page, "_toolbar", None)
        canvas = getattr(page, "_canvas", None)
        if lay is None or toolbar is None or canvas is None:
            return

        i_toolbar = lay.indexOf(toolbar)
        i_canvas = lay.indexOf(canvas)
        if i_toolbar == -1 or i_canvas == -1:
            return

        if i_toolbar > i_canvas:  # déjà en bas
            return

        lay.removeWidget(toolbar)
        lay.removeWidget(canvas)
        lay.addWidget(canvas, 1)
        lay.addWidget(toolbar, 0)

        toolbar.show()
        canvas.show()

    # ----- Création d’onglet -----
    def creer_onglet(chemin: str, contenu: str):
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        fig, ax = plt.subplots()
        fig.patch.set_facecolor('#1E1E1E')
        ax.set_facecolor('#1E1E1E')

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        toolbar = NavigationToolbar(canvas, page)
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #1E1E1E;
                border: none;
                spacing: 6px;
            }
            QToolButton {
                background-color: #AEBAB2;
                color: white;
                border-radius: 6px;
                padding: 4px;
            }
            QToolButton:hover { background-color: #3A3A3A; }
        """)

        page_layout.addWidget(toolbar, 0)
        page_layout.addWidget(canvas, 1)

        page._layout = page_layout
        page._fig = fig
        page._ax = ax
        page._canvas = canvas
        page._toolbar = toolbar
        page._data_text = contenu
        page._path = chemin

        nom = chemin.split("/")[-1].split("\\")[-1]
        tabs.addTab(page, nom)

        fichier_label.setText(f"{tabs.count()} fichier(s) chargé(s)")

    # ----- Ajout fichiers -----
    def ajouter_fichier():
        chemins, _ = QFileDialog.getOpenFileNames(
            fenêtre,
            "Choisir un ou plusieurs fichiers",
            "",
            "Fichiers texte (*.txt *.dat *.csv);;Tous les fichiers (*)"
        )
        if not chemins:
            return

        ok = 0
        for chemin in chemins:
            contenu = lire_fichier(chemin)
            if contenu is None:
                continue
            creer_onglet(chemin, contenu)
            ok += 1

        if ok == 0:
            erreur_fichiers()

    bouton_ajouter.clicked.connect(ajouter_fichier)

    # ----- Boutons -----
    bouton_de_validation = QPushButton("Valider les données")
    bouton_de_validation.setStyleSheet("""
        QPushButton {
            background-color: #6C63FF;
            border-radius: 12px;
            padding: 12px;
            color: white;
            font-size: 14px;
        }
        QPushButton:hover { background-color: #574fd6; }
    """)
    bouton_de_validation.setFocusPolicy(Qt.NoFocus)
    cadre_layout.addWidget(bouton_de_validation)

    bouton_popout = QPushButton("afficher les graphes en plein écran ")
    bouton_popout.setStyleSheet("""
        QPushButton {
            background-color: #00C896;
            border-radius: 12px;
            padding: 12px;
            color: white;
            font-size: 14px;
        }
        QPushButton:hover { background-color: #00a87d; }
    """)
    bouton_popout.setFocusPolicy(Qt.NoFocus)
    cadre_layout.addWidget(bouton_popout)

    # ----- Boutons analyse -----
    t12_button = QPushButton("Calculer t1/2 pour cette série de données ")
    ea_button = QPushButton("Tracer Ea pour cette série de données ")
    déterminer_ordre = QPushButton("Déterminer l'ordre pour cette série de données")
    resume_button = QPushButton("Résumé de vos valeurs")

    for bouton in [t12_button, ea_button, déterminer_ordre, resume_button]:
        bouton.setEnabled(False)
        bouton.setStyleSheet("""
            QPushButton {
                background-color: #444444;
                border-radius: 12px;
                padding: 12px;
                color: #888888;
                font-size: 14px;
                outline: none;
            }
        """)
        cadre_layout.addWidget(bouton)

    def activer_boutons_analyse():
        for btn in [t12_button, ea_button, déterminer_ordre, resume_button]:
            btn.setEnabled(True)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #00C896;
                    border-radius: 12px;
                    padding: 12px;
                    color: white;
                    font-size: 14px;
                    outline: none;
                }
                QPushButton:hover { background-color: #00a87d; }
            """)

    # ---------- TRACÉ : 1 onglet (points + trendline) ----------
    def tracer_un_onglet(page):
        temps = lire_temperatures()
        if temps is None:
            erreur_temperature()
            return False

        parsed = parser_donnees(getattr(page, "_data_text", ""))
        if parsed is None:
            return False

        # toolbar en bas à la validation
        mettre_toolbar_en_bas(page)

        t, X = parsed
        ax, fig, canvas = page._ax, page._fig, page._canvas

        ax.clear()
        ax.set_xscale("log")

        ax.set_facecolor('#1E1E1E')
        fig.patch.set_facecolor('#1E1E1E')

        colors = ["#FF6B6B", "#48bfb8", "#FFD93D", "#845EC2"]

        for i in range(4):
            ax.plot(
                t, X[:, i],
                linestyle="None",
                marker="o",
                markersize=6,
                markerfacecolor=colors[i],
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=f"{temps[i]:g} K"
            )

        Xm = trendline_modele(t, X)
        if Xm is not None:
            for i in range(4):
                ax.plot(t, Xm[:, i], linewidth=2.2, alpha=0.95)

        ax.grid(alpha=0.2)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.set_xlabel("Temps (s)", color="white")
        ax.set_ylabel("Avancement (mol)", color="white")

        leg = ax.legend(loc="best")
        leg.get_frame().set_facecolor("#1E1E1E")
        leg.get_frame().set_edgecolor("white")
        for txt in leg.get_texts():
            txt.set_color("white")

        fig.tight_layout()
        canvas.draw_idle()
        return True

    # ---------- TRACÉ : TOUS les onglets ----------
    def tracer_tous_les_onglets():
        if tabs.count() == 0:
            popup_blanc("Ajoutez un ou plusieurs fichiers.", "Info")
            return

        current = tabs.currentIndex()

        for i in range(tabs.count()):
            page = tabs.widget(i)
            if tracer_un_onglet(page) is False:
                erreur_format()
                return

        tabs.setCurrentIndex(current)
        redraw_current_tab()
        activer_boutons_analyse()

    bouton_de_validation.clicked.connect(tracer_tous_les_onglets)

    # ============================================================
    def page_courante():
        p = tabs.currentWidget()
        if p is None:
            popup_blanc("Aucun onglet sélectionné.", "Info")
            return None
        return p

    def calculer_fit_pour_page(p):
        temps = lire_temperatures()
        if temps is None:
            erreur_temperature()
            return None

        parsed = parser_donnees(getattr(p, "_data_text", ""))
        if parsed is None:
            erreur_format()
            return None

        t, X = parsed
        Xnorm, _ = normaliser_X(X)
        try:
            n_star, k_opt = fit_n_et_k(t, Xnorm)
        except Exception as e:
            popup_blanc(f"Fit impossible.\nDétail : {e}", "Erreur")
            return None
        return temps, n_star, k_opt

    def action_ordre():
        p = page_courante()
        if p is None:
            return
        out = calculer_fit_pour_page(p)
        if out is None:
            return
        _, n_star, _ = out
        popup_blanc(f"Ordre ajusté : n = {float(n_star):.6g}", "Résultat")

    def action_t12():
        p = page_courante()
        if p is None:
            return
        out = calculer_fit_pour_page(p)
        if out is None:
            return
        temps, n_star, k_opt = out
        t12 = t_demi(n_star, k_opt)
        msg = "t1/2 \n"
        for Ti, val in zip(temps, t12):
            msg += f"{Ti:g} K : {val:.6g}\n"
        popup_blanc(msg, "Résultat")

    def action_ea():
        p = page_courante()
        if p is None:
            return
        out = calculer_fit_pour_page(p)
        if out is None:
            return
        temps, n_star, k_opt = out
        popup_arrhenius(temps, k_opt, n_star)

    déterminer_ordre.clicked.connect(action_ordre)
    t12_button.clicked.connect(action_t12)
    ea_button.clicked.connect(action_ea)

    # =======================résumé =====================================
    resume_window = {"w": None}

    def ouvrir_resume_fenetre():
        temps = lire_temperatures()
        if temps is None:
            erreur_temperature()
            return
        if tabs.count() == 0:
            popup_blanc("Ajoutez un ou plusieurs fichiers.", "Info")
            return

        if resume_window["w"] is not None and resume_window["w"].isVisible():
            resume_window["w"].raise_()
            resume_window["w"].activateWindow()
            return

        def build(ui_parent):
            tabs_resume = QTabWidget(ui_parent)
            tabs_resume.setStyleSheet("""
                QTabWidget::pane { border: 0; }
                QTabBar::tab {
                    background: #2C2C2C;
                    color: #BBBBBB;
                    padding: 10px 14px;
                    border-top-left-radius: 10px;
                    border-top-right-radius: 10px;
                    margin-right: 6px;
                }
                QTabBar::tab:selected { background: #3A3A3A; color: white; }
            """)
            tabs_resume.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            ui_parent._main.addWidget(tabs_resume, 1)

            def plot_cinetique(ax, t, X, temps_local):
                ax.clear()
                ax.set_xscale("log")
                ax.set_facecolor("#1E1E1E")
                colors = ["#FF6B6B", "#48bfb8", "#FFD93D", "#845EC2"]

                for i in range(4):
                    ax.plot(
                        t, X[:, i],
                        linestyle="None",
                        marker="o",
                        markersize=6,
                        markerfacecolor=colors[i],
                        markeredgecolor="white",
                        markeredgewidth=0.8,
                        label=f"{temps_local[i]:g} K"
                    )

                Xm = trendline_modele(t, X)
                if Xm is not None:
                    for i in range(4):
                        ax.plot(t, Xm[:, i], linewidth=2.2, alpha=0.95)

                ax.grid(alpha=0.2)
                ax.tick_params(colors="white")
                ax.spines["bottom"].set_color("white")
                ax.spines["left"].set_color("white")
                ax.set_xlabel("Temps (s)", color="white")
                ax.set_ylabel("Avancement (mol)", color="white")

                leg = ax.legend(loc="best")
                leg.get_frame().set_facecolor("#1E1E1E")
                leg.get_frame().set_edgecolor("white")
                for txt in leg.get_texts():
                    txt.set_color("white")

            def plot_arrhenius(ax, temps_local, k_opt, n_star):
                T = np.asarray(temps_local, float)
                k = np.asarray(k_opt, float)
                k = np.clip(k, 1e-300, None)

                x = 1.0 / T
                y = np.log(k)
                a, b = np.polyfit(x, y, 1)
                Ea = -a * 8.314 / 1000.0

                ax.clear()
                ax.set_facecolor("#1E1E1E")
                ax.plot(x, y, linestyle="None", marker="o", markersize=7,
                        markerfacecolor="white", markeredgecolor="white")

                xfit = np.linspace(x.min() * 0.98, x.max() * 1.02, 200)
                yfit = a * xfit + b
                ax.plot(xfit, yfit, linewidth=2.4)

                ax.grid(alpha=0.2)
                ax.tick_params(colors="white")
                ax.spines["bottom"].set_color("white")
                ax.spines["left"].set_color("white")
                ax.set_xlabel("1 / T (K⁻¹)", color="white")
                ax.set_ylabel("ln(k)", color="white")
                ax.set_title(f"Ea = {Ea:.6g} kJ/mol  |  n = {float(n_star):.6g}",
                             color="white", pad=10)
                return Ea

            for i in range(tabs.count()):
                page_src = tabs.widget(i)
                nom = tabs.tabText(i)

                parsed = parser_donnees(getattr(page_src, "_data_text", ""))
                if parsed is None:
                    erreur_format()
                    return
                t, X = parsed

                Xnorm, _ = normaliser_X(X)
                try:
                    n_star, k_opt = fit_n_et_k(t, Xnorm)
                except Exception as e:
                    popup_blanc(f"Fit impossible pour {nom}.\nDétail : {e}", "Erreur")
                    return

                t12 = t_demi(n_star, k_opt)

                page = QWidget()
                lay = QVBoxLayout(page)
                lay.setContentsMargins(0, 0, 0, 0)
                lay.setSpacing(10)

                graphs = QWidget()
                gl = QHBoxLayout(graphs)
                gl.setContentsMargins(0, 0, 0, 0)
                gl.setSpacing(10)

                fig1, ax1 = plt.subplots()
                fig1.patch.set_facecolor("#1E1E1E")
                ax1.set_facecolor("#1E1E1E")
                canvas1 = FigureCanvas(fig1)
                toolbar1 = NavigationToolbar(canvas1, ui_parent)
                toolbar1.setStyleSheet("""
                    QToolBar { background-color: #1E1E1E; border: none; spacing: 6px; }
                    QToolButton { background-color: #AEBAB2; color: white; border-radius: 6px; padding: 4px; }
                    QToolButton:hover { background-color: #3A3A3A; }
                """)

                fig2, ax2 = plt.subplots()
                fig2.patch.set_facecolor("#1E1E1E")
                ax2.set_facecolor("#1E1E1E")
                canvas2 = FigureCanvas(fig2)
                toolbar2 = NavigationToolbar(canvas2, ui_parent)
                toolbar2.setStyleSheet(toolbar1.styleSheet())

                block1 = QWidget()
                b1 = QVBoxLayout(block1)
                b1.setContentsMargins(0, 0, 0, 0)
                b1.setSpacing(6)
                b1.addWidget(canvas1, 1)
                b1.addWidget(toolbar1, 0)

                block2 = QWidget()
                b2 = QVBoxLayout(block2)
                b2.setContentsMargins(0, 0, 0, 0)
                b2.setSpacing(6)
                b2.addWidget(canvas2, 1)
                b2.addWidget(toolbar2, 0)

                gl.addWidget(block1, 1)
                gl.addWidget(block2, 1)
                lay.addWidget(graphs, 1)

                plot_cinetique(ax1, t, X, temps)
                Ea = plot_arrhenius(ax2, temps, k_opt, n_star)

                fig1.subplots_adjust(left=0.15, right=0.98, bottom=0.18, top=0.90)
                fig2.subplots_adjust(left=0.15, right=0.98, bottom=0.18, top=0.90)

                canvas1.draw_idle()
                canvas2.draw_idle()

                msg = f"Fichier : {nom}\n\n"
                msg += f"Ordre ajusté : n = {float(n_star):.6g}\n"
                msg += f"Ea = {Ea:.6g} kJ/mol\n\n"
                msg += "t1/2:\n"
                for Ti, val in zip(temps, t12):
                    msg += f"{Ti:g} K : {val:.6g}\n"
                msg += "\nConstantes k optimisées :\n"
                for Ti, ki in zip(temps, k_opt):
                    msg += f"{Ti:g} K : {ki:.8g}\n"

                label = QLabel(msg)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                label.setStyleSheet("color: white; font-size: 14px;")
                lay.addWidget(label, 0)

                tabs_resume.addTab(page, nom)

        def on_close():
            resume_window["w"] = None

        resume_window["w"] = FenetreResume(build_callback=build, on_close_callback=on_close)

    resume_button.clicked.connect(ouvrir_resume_fenetre)

    # ============================================================
    #  graphes
    graph_window = {"w": None}

    def remettre_tabs_dans_principal():
        tabs.setParent(tabs_container)
        tabs_container_layout.addWidget(tabs, 1)
        tabs.show()
        redraw_current_tab()
        graph_window["w"] = None

    def ouvrir_graphes_fenetre():
        if tabs.count() == 0:
            popup_blanc("Ajoute d’abord un fichier.", "Info")
            return

        if graph_window["w"] is not None and graph_window["w"].isVisible():
            graph_window["w"].raise_()
            graph_window["w"].activateWindow()
            return

        w = FenetreGraphes(tabs, remettre_tabs_dans_principal)
        graph_window["w"] = w

    bouton_popout.clicked.connect(ouvrir_graphes_fenetre)

    # ----- Final -----
    principal_layout.addWidget(cadre)
    fenêtre.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    principal()