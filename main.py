import os
import sys
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from skimage import io, filters, measure, segmentation
from skimage.filters import threshold_multiotsu
from skimage.feature import peak_local_max
from scipy.ndimage import (gaussian_filter, binary_opening, binary_closing,
                           label, generate_binary_structure,
                           distance_transform_edt, sobel as ndi_sobel)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QSlider, QLabel, QFileDialog,
                             QGroupBox, QComboBox, QMessageBox, QCheckBox, QDialog,
                             QScrollArea, QFrame)
from PyQt6.QtCore import Qt

try:
    from sklearn.cluster import MiniBatchKMeans

    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


class AedesAtlasApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Atlas 3D Interativo - Aedes aegypti")
        self.setGeometry(100, 100, 1300, 850)

        self.volume_data = None
        self.volume_smooth = None

        self.kmeans_labels = None
        self.kmeans_k = 0
        self.watershed_labels = None

        self.apply_stylesheet()
        self.init_ui()

    def apply_stylesheet(self):
        dark_stylesheet = """
        QMainWindow {
            background-color: #1e1e1e;
        }
        QWidget {
            color: #e0e0e0;
            font-family: "Segoe UI", Arial, sans-serif;
            font-size: 10pt;
        }
        QGroupBox {
            background-color: #252526;
            border: 1px solid #3e3e42;
            border-radius: 6px;
            margin-top: 14px;
            padding-top: 10px;
            padding-bottom: 5px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: #4fa8ff;
            font-weight: bold;
        }
        QPushButton {
            background-color: #0e639c;
            color: #ffffff;
            border: none;
            border-radius: 4px;
            padding: 8px 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1177bb;
        }
        QPushButton:pressed {
            background-color: #094771;
        }
        QPushButton:disabled {
            background-color: #3e3e42;
            color: #7a7a7a;
        }
        QComboBox {
            background-color: #333337;
            border: 1px solid #3e3e42;
            border-radius: 4px;
            padding: 4px;
            color: #e0e0e0;
        }
        QComboBox:drop-down {
            border-left: 1px solid #3e3e42;
        }
        QSlider::groove:horizontal {
            border: 1px solid #3e3e42;
            height: 6px;
            background: #1e1e1e;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #4fa8ff;
            border: 1px solid #4fa8ff;
            width: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }
        QSlider::handle:horizontal:hover {
            background: #78bfff;
        }
        QCheckBox {
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid #3e3e42;
            border-radius: 3px;
            background: #333337;
        }
        QCheckBox::indicator:checked {
            background: #0e639c;
            border: 1px solid #0e639c;
        }
        QScrollArea {
            border: none;
            background-color: #1e1e1e;
        }
        QScrollBar:vertical {
            border: none;
            background: #1e1e1e;
            width: 10px;
            margin: 0px 0px 0px 0px;
        }
        QScrollBar::handle:vertical {
            background: #424242;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical:hover {
            background: #4f4f4f;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        """
        self.setStyleSheet(dark_stylesheet)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # PAINEL ESQUERDO
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(380)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 15, 5)

        # --- 1. Aquisição e Configuração ---
        group_io = QGroupBox("1. Aquisição e Configuração")
        io_layout = QVBoxLayout()
        self.btn_load = QPushButton("Carregar Pasta com Fatias (.bmp/.tif)")
        self.btn_load.clicked.connect(self.load_directory)
        io_layout.addWidget(self.btn_load)

        io_layout.addWidget(QLabel("Esquema de Cores (Acessibilidade):"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItem("Turbo (Contraste Padrão)", "turbo")
        self.combo_cmap.addItem("Cividis (Deuteranopia Seguro)", "cividis")
        self.combo_cmap.addItem("Viridis (Deuteranopia Seguro)", "viridis")
        self.combo_cmap.addItem("Plasma", "plasma")
        io_layout.addWidget(self.combo_cmap)

        group_io.setLayout(io_layout)

        # --- 2. Renderização Volumétrica ---
        group_vol = QGroupBox("2. Renderização Volumétrica")
        vol_layout = QVBoxLayout()
        self.btn_render_vol = QPushButton("Exibir Volume Completo")
        self.btn_render_vol.clicked.connect(self.show_volume_rendering)
        vol_layout.addWidget(self.btn_render_vol)
        group_vol.setLayout(vol_layout)

        # --- 3. Isolamento Paramétrico ---
        group_iso = QGroupBox("3. Isolamento de Estruturas")
        iso_layout = QVBoxLayout()

        self.btn_histogram = QPushButton("Analisar Densidades (Otsu)")
        self.btn_histogram.clicked.connect(self.show_density_histogram)
        iso_layout.addWidget(self.btn_histogram)

        iso_layout.addWidget(QLabel("Alvo Anatômico Predefinido:"))
        self.combo_organs = QComboBox()
        self.combo_organs.addItems(["Selecione um Tecido...",
                                    "Exoesqueleto / Músculos Densos",
                                    "Tecidos Moles / Intestino",
                                    "Ajuste Manual"])
        self.combo_organs.currentIndexChanged.connect(self.preset_threshold)
        iso_layout.addWidget(self.combo_organs)

        self.lbl_thresh_min = QLabel("Densidade Mínima: 60")
        self.slider_thresh_min = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh_min.setRange(1, 254)
        self.slider_thresh_min.setValue(60)
        self.slider_thresh_min.valueChanged.connect(self.update_thresh_label)

        self.lbl_thresh_max = QLabel("Densidade Máxima: 140")
        self.slider_thresh_max = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh_max.setRange(1, 254)
        self.slider_thresh_max.setValue(140)
        self.slider_thresh_max.valueChanged.connect(self.update_thresh_label)

        self.lbl_denoise = QLabel("Limpeza Morfológica: 1")
        self.slider_denoise = QSlider(Qt.Orientation.Horizontal)
        self.slider_denoise.setRange(0, 3)
        self.slider_denoise.setValue(1)
        self.slider_denoise.valueChanged.connect(
            lambda v: self.lbl_denoise.setText(f"Limpeza Morfológica: {v}")
        )

        iso_layout.addWidget(self.lbl_thresh_min)
        iso_layout.addWidget(self.slider_thresh_min)
        iso_layout.addWidget(self.lbl_thresh_max)
        iso_layout.addWidget(self.slider_thresh_max)
        iso_layout.addWidget(self.lbl_denoise)
        iso_layout.addWidget(self.slider_denoise)

        self.btn_extract = QPushButton("Gerar Malha Filtrada")
        self.btn_extract.clicked.connect(self.extract_and_show_mesh)
        iso_layout.addWidget(self.btn_extract)

        group_iso.setLayout(iso_layout)

        # --- 4. Segmentação Avançada (K-Means / Watershed) ---
        group_adv = QGroupBox("4. Segmentação Avançada")
        adv_layout = QVBoxLayout()

        # K-Means Sub-group
        adv_layout.addWidget(QLabel("<b>K-Means (Densidade + Sobel)</b>"))
        self.lbl_kmeans_k = QLabel("Nº de Clusters (K): 4")
        self.slider_kmeans_k = QSlider(Qt.Orientation.Horizontal)
        self.slider_kmeans_k.setRange(2, 8)
        self.slider_kmeans_k.setValue(4)
        self.slider_kmeans_k.valueChanged.connect(
            lambda v: self.lbl_kmeans_k.setText(f"Nº de Clusters (K): {v}")
        )

        km_btn_layout = QHBoxLayout()
        self.btn_run_kmeans = QPushButton("Executar")
        self.btn_run_kmeans.clicked.connect(self.run_kmeans)
        self.btn_isolate_kmeans = QPushButton("Isolar")
        self.btn_isolate_kmeans.clicked.connect(self.isolate_kmeans_cluster)
        km_btn_layout.addWidget(self.btn_run_kmeans)
        km_btn_layout.addWidget(self.btn_isolate_kmeans)

        self.combo_kmeans_cluster = QComboBox()

        adv_layout.addWidget(self.lbl_kmeans_k)
        adv_layout.addWidget(self.slider_kmeans_k)
        adv_layout.addLayout(km_btn_layout)
        adv_layout.addWidget(self.combo_kmeans_cluster)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #3e3e42;")
        adv_layout.addWidget(line)

        # Watershed Sub-group
        adv_layout.addWidget(QLabel("<b>Watershed (Separação Topológica)</b>"))
        self.combo_ws_relevo = QComboBox()
        self.combo_ws_relevo.addItems([
            "Gradiente / Sobel",
            "Distância Euclidiana",
        ])
        adv_layout.addWidget(self.combo_ws_relevo)

        self.lbl_ws_mindist = QLabel("Distância Mín. entre Sementes: 5")
        self.slider_ws_mindist = QSlider(Qt.Orientation.Horizontal)
        self.slider_ws_mindist.setRange(2, 20)
        self.slider_ws_mindist.setValue(5)
        self.slider_ws_mindist.valueChanged.connect(
            lambda v: self.lbl_ws_mindist.setText(f"Distância Mín. entre Sementes: {v}")
        )

        ws_btn_layout = QHBoxLayout()
        self.btn_run_watershed = QPushButton("Executar")
        self.btn_run_watershed.clicked.connect(self.run_watershed)
        self.btn_isolate_ws = QPushButton("Isolar Região")
        self.btn_isolate_ws.clicked.connect(self.isolate_watershed_region)
        ws_btn_layout.addWidget(self.btn_run_watershed)
        ws_btn_layout.addWidget(self.btn_isolate_ws)

        self.combo_ws_region = QComboBox()

        self.btn_view_all_ws = QPushButton("Visualizar Todas as Regiões")
        self.btn_view_all_ws.clicked.connect(self.view_all_watershed_regions)

        adv_layout.addWidget(self.lbl_ws_mindist)
        adv_layout.addWidget(self.slider_ws_mindist)
        adv_layout.addLayout(ws_btn_layout)
        adv_layout.addWidget(self.combo_ws_region)
        adv_layout.addWidget(self.btn_view_all_ws)

        group_adv.setLayout(adv_layout)

        # --- 5. Controles de Cena ---
        group_scene = QGroupBox("5. Cena Interativa")
        scene_layout = QVBoxLayout()

        self.chk_carcaca = QCheckBox("Mostrar Carcaça (corpo fantasma) ao isolar")
        self.chk_carcaca.setChecked(True)
        self.chk_carcaca.setToolTip(
            "Ao isolar uma estrutura — por limiar de densidade, cluster do "
            "K-Means ou região do Watershed — exibe também o corpo/segmentação "
            "completa como uma casca translúcida de referência, com apenas a "
            "parte selecionada colorida em destaque."
        )

        self.chk_clip = QCheckBox("Ativar Plano de Corte (*Clipping*)")
        self.chk_clip.setChecked(False)
        self.btn_clear = QPushButton("Limpar Visualizador")
        self.btn_clear.clicked.connect(self.clear_scene)
        scene_layout.addWidget(self.chk_carcaca)
        scene_layout.addWidget(self.chk_clip)
        scene_layout.addWidget(self.btn_clear)
        group_scene.setLayout(scene_layout)

        # Populate control layout
        control_layout.addWidget(group_io)
        control_layout.addWidget(group_vol)
        control_layout.addWidget(group_iso)
        control_layout.addWidget(group_adv)
        control_layout.addWidget(group_scene)
        control_layout.addStretch()

        scroll_area.setWidget(control_panel)

        # PAINEL DIREITO
        self.plotter = QtInteractor(self)
        self.plotter.add_axes()
        self.plotter.add_text("Aguardando importação de dados...", font_size=10, name="status")

        main_layout.addWidget(scroll_area)
        main_layout.addWidget(self.plotter.interactor)

    # FUNÇÕES LÓGICAS E MATEMÁTICAS
    def load_directory(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Selecione a pasta com os .bmp")
        if not folder_path:
            return

        try:
            files = sorted([f for f in os.listdir(folder_path) if f.endswith('.bmp') or f.endswith('.tif')])
            if not files:
                QMessageBox.warning(self, "Erro", "Nenhuma imagem encontrada na pasta.")
                return

            self.plotter.add_text("Carregando matriz tridimensional...", name="status")
            self.plotter.update()
            QApplication.processEvents()

            passo = 2
            slices = []
            expected_shape = None

            for file in files[::passo]:
                filepath = os.path.join(folder_path, file)
                img = io.imread(filepath, as_gray=True)
                img_reduzida = img[::passo, ::passo]

                if expected_shape is None:
                    expected_shape = img_reduzida.shape
                    slices.append(img_reduzida)
                elif img_reduzida.shape == expected_shape:
                    slices.append(img_reduzida)

            self.volume_data = np.stack(slices, axis=0)

            if self.volume_data.max() <= 1.0:
                self.volume_data = (self.volume_data * 255).astype(np.uint8)

            self.plotter.add_text("Aplicando suavização gaussiana...", name="status")
            self.plotter.update()
            QApplication.processEvents()

            self.volume_smooth = gaussian_filter(
                self.volume_data.astype(np.float32), sigma=0.6
            )

            self.plotter.add_text(f"Volume carregado: {len(slices)} fatias.", name="status")
            QMessageBox.information(self, "Sucesso", "Volume carregado com sucesso!")

        except MemoryError:
            QMessageBox.critical(self, "Falta de Memória",
                                 "O volume excede a memória RAM. Aumente o valor de 'passo'.")
        except Exception as e:
            QMessageBox.critical(self, "Erro Crítico", f"Erro ao carregar as imagens:\n{str(e)}")

    def show_volume_rendering(self):
        if self.volume_smooth is None:
            QMessageBox.warning(self, "Erro", "Carregue os dados primeiro!")
            return

        self.plotter.add_text("Calculando Ray Casting...", name="status", color="white")
        self.plotter.update()
        QApplication.processEvents()

        self.clear_scene()
        self.plotter.set_background("#1e1e1e")

        try:
            grid = pv.wrap(self.volume_data)
            selected_cmap = self.combo_cmap.currentData()

            if self.chk_clip.isChecked():
                self.plotter.add_volume_clip_plane(
                    grid,
                    cmap=selected_cmap,
                    opacity="linear",
                    shade=False
                )
                self.plotter.add_text("Modo: Renderização Volumétrica (Com Corte)", name="status", color="white")
            else:
                self.plotter.add_volume(
                    grid,
                    cmap=selected_cmap,
                    opacity="linear",
                    shade=False,
                    name="colored_vol"
                )
                self.plotter.add_text("Modo: Renderização Volumétrica", name="status", color="white")

            self.plotter.reset_camera()

        except Exception as e:
            QMessageBox.critical(self, "Erro na Renderização", f"Falha ao gerar o volume:\n{str(e)}")

    def update_thresh_label(self, _value=None):
        self.lbl_thresh_min.setText(f"Densidade Mínima: {self.slider_thresh_min.value()}")
        self.lbl_thresh_max.setText(f"Densidade Máxima: {self.slider_thresh_max.value()}")

    def preset_threshold(self):
        idx = self.combo_organs.currentIndex()
        if idx == 1:
            self.slider_thresh_min.setValue(160)
            self.slider_thresh_max.setValue(254)
        elif idx == 2:
            self.slider_thresh_min.setValue(50)
            self.slider_thresh_max.setValue(140)
        self.update_thresh_label()

    def show_density_histogram(self):
        if self.volume_smooth is None:
            QMessageBox.warning(self, "Erro", "Carregue os dados primeiro!")
            return

        dados = self.volume_smooth[self.volume_smooth > 15].ravel()

        if dados.size == 0:
            QMessageBox.warning(self, "Aviso", "Volume parece vazio (só fundo/ar).")
            return

        try:
            cortes = threshold_multiotsu(dados, classes=4)
        except Exception:
            cortes = threshold_multiotsu(dados, classes=3)

        dialog = QDialog(self)
        dialog.setWindowTitle("Distribuição de Densidades")
        dialog.resize(720, 520)
        dialog.setStyleSheet("background-color: #1e1e1e; color: #e0e0e0;")
        layout = QVBoxLayout(dialog)

        fig = Figure(figsize=(7, 4.5), facecolor='#1e1e1e')
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='#e0e0e0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#3e3e42')

        ax.hist(dados, bins=150, color="#4fa8ff", log=True)
        for c in cortes:
            ax.axvline(c, color="#f44336", linestyle="--")
            ax.text(c, ax.get_ylim()[1] * 0.85, f"{c:.0f}", color="#f44336", rotation=90)

        ax.set_xlabel("Densidade (0-255)", color='#e0e0e0')
        ax.set_ylabel("Nº de voxels (escala log)", color='#e0e0e0')
        ax.set_title("Histograma de Densidade do Tecido", color='#e0e0e0')
        layout.addWidget(canvas)

        texto_cortes = ", ".join(f"{c:.0f}" for c in cortes)
        lbl = QLabel(f"Cortes sugeridos (Otsu): {texto_cortes}")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        dialog.exec()

    def extract_and_show_mesh(self):
        if self.volume_smooth is None:
            QMessageBox.warning(self, "Erro", "Carregue os dados primeiro!")
            return

        thresh_min = self.slider_thresh_min.value()
        thresh_max = self.slider_thresh_max.value()
        denoise_level = self.slider_denoise.value()

        if thresh_min >= thresh_max:
            QMessageBox.warning(self, "Erro", "A densidade mínima deve ser menor que a máxima.")
            return

        self.plotter.add_text(f"Extraindo Órgãos...", name="status", color="white")
        self.plotter.update()
        QApplication.processEvents()

        try:
            mask = (self.volume_smooth > thresh_min) & (self.volume_smooth < thresh_max)
            if not mask.any():
                QMessageBox.warning(self, "Aviso", "Nenhum tecido encontrado nesta faixa.")
                return

            # Carcaça
            self._render_mask_as_mesh(
                mask, denoise_level, f"Estruturas ({thresh_min}-{thresh_max})",
                context_mask=(self.volume_smooth > 15), context_denoise=1
            )

        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha na malha:\n{str(e)}")

    def _build_mesh_from_mask(self, mask, denoise_level):
        if not mask.any():
            QMessageBox.warning(self, "Aviso", "Máscara vazia.")
            return None

        struct_leve = generate_binary_structure(3, 1)
        if denoise_level > 0:
            mask = binary_opening(mask, structure=struct_leve, iterations=1)
            mask = binary_closing(mask, structure=struct_leve, iterations=1)

        struct = generate_binary_structure(3, 2)
        labeled, n_labels = label(mask, structure=struct)
        if n_labels == 0:
            QMessageBox.warning(self, "Aviso", "Nada sobrou após limpeza.")
            return None

        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        if sizes.max() == 0:
            QMessageBox.warning(self, "Aviso", "Nada sobrou após limpeza.")
            return None

        min_voxels = {0: 1, 1: 4, 2: 10, 3: 25}.get(denoise_level, 4)
        labels_validos = np.where(sizes >= min_voxels)[0]
        mask_limpa = np.isin(labeled, labels_validos)

        if not mask_limpa.any():
            QMessageBox.warning(self, "Aviso", "Nada passou pelo filtro de tamanho.")
            return None

        verts, faces, normals, values = measure.marching_cubes(
            mask_limpa.astype(np.float32), level=0.5
        )
        faces_pv = np.hstack(
            [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
        ).astype(np.int64)

        mesh = pv.PolyData(verts, faces_pv)
        mesh = mesh.smooth(n_iter=60, relaxation_factor=0.15)
        mesh = mesh.compute_normals(auto_orient_normals=True, inplace=False)
        return mesh

    def _add_carcaca_ghost(self, mask_corpo, denoise_level=0):
        mesh_corpo = self._build_mesh_from_mask(mask_corpo, denoise_level)
        if mesh_corpo is not None:
            self.plotter.add_mesh(
                mesh_corpo,
                color="#aab0b8",
                opacity=0.10,
                smooth_shading=True,
                specular=0.0,
                name="carcaca",
                pickable=False,
            )

    def _render_mask_as_mesh(self, mask, denoise_level, label_text, color_label_volume=None,
                              context_mask=None, context_denoise=0):
        mesh = self._build_mesh_from_mask(mask, denoise_level)
        if mesh is None:
            return

        selected_cmap = self.combo_cmap.currentData()
        mostrar_carcaca = context_mask is not None and self.chk_carcaca.isChecked()

        scalars_ativo = cmap_ativo = clim_ativo = None

        if color_label_volume is not None:
            coords = np.clip(
                np.round(mesh.points).astype(int), 0, np.array(self.volume_data.shape) - 1
            )
            rotulo_por_vertice = color_label_volume[
                coords[:, 0], coords[:, 1], coords[:, 2]
            ]
            mesh["Regiao"] = rotulo_por_vertice.astype(np.float32)
            scalars_ativo = "Regiao"
            cmap_ativo = "tab20"
            clim_ativo = [0, max(1, int(color_label_volume.max()))]
        else:
            coords = np.clip(
                np.round(mesh.points).astype(int), 0, np.array(self.volume_data.shape) - 1
            )
            densidade_por_vertice = self.volume_data[
                coords[:, 0], coords[:, 1], coords[:, 2]
            ]
            mesh["Densidade"] = densidade_por_vertice
            scalars_ativo = "Densidade"
            cmap_ativo = selected_cmap
            clim_ativo = [0, 255]

        self.clear_scene()
        self.plotter.set_background("#1e1e1e")

        if mostrar_carcaca:
            self._add_carcaca_ghost(context_mask, context_denoise)

        kwargs_cor = {"scalars": scalars_ativo, "cmap": cmap_ativo, "clim": clim_ativo}
        specular = 0.5
        sufixo = " (Com Corte)" if self.chk_clip.isChecked() else ""

        if self.chk_clip.isChecked():
            self.plotter.add_mesh_clip_plane(
                mesh, smooth_shading=True, specular=specular, **kwargs_cor
            )
        else:
            self.plotter.add_mesh(
                mesh, smooth_shading=True, specular=specular, name="iso_mesh", **kwargs_cor
            )

        self.plotter.add_text(f"Modo: {label_text}{sufixo}", name="status", color="white")
        self.plotter.reset_camera()

    def _gradient_magnitude(self, volume):
        gx = ndi_sobel(volume, axis=0)
        gy = ndi_sobel(volume, axis=1)
        gz = ndi_sobel(volume, axis=2)
        return np.sqrt(
            gx.astype(np.float64) ** 2 + gy.astype(np.float64) ** 2 + gz.astype(np.float64) ** 2
        )

    def run_kmeans(self):
        if self.volume_smooth is None:
            QMessageBox.warning(self, "Erro", "Carregue os dados primeiro!")
            return
        if not SKLEARN_OK:
            QMessageBox.critical(self, "Dependência", "Pacote scikit-learn não instalado.")
            return

        k = self.slider_kmeans_k.value()
        self.plotter.add_text(f"Calculando K-Means (K={k})...", name="status", color="white")
        self.plotter.update()
        QApplication.processEvents()

        try:
            foreground = self.volume_smooth > 15
            if not foreground.any():
                QMessageBox.warning(self, "Aviso", "Volume parece vazio.")
                return

            grad = self._gradient_magnitude(self.volume_smooth)
            dens_fg = self.volume_smooth[foreground].astype(np.float64)
            grad_fg = grad[foreground]

            dens_norm = (dens_fg - dens_fg.min()) / (np.ptp(dens_fg) + 1e-8)
            grad_norm = (grad_fg - grad_fg.min()) / (np.ptp(grad_fg) + 1e-8)
            features = np.column_stack([dens_norm, grad_norm])

            km = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=0, batch_size=4096)
            cluster_ids = km.fit_predict(features)

            label_volume = np.full(self.volume_smooth.shape, -1, dtype=np.int32)
            label_volume[foreground] = cluster_ids
            self.kmeans_labels = label_volume
            self.kmeans_k = k

            ordem = sorted(
                range(k),
                key=lambda c: dens_fg[cluster_ids == c].mean() if np.any(cluster_ids == c) else 0
            )

            self.combo_kmeans_cluster.clear()
            for c in ordem:
                n_voxels = int(np.sum(cluster_ids == c))
                dens_media = dens_fg[cluster_ids == c].mean() if n_voxels > 0 else 0
                self.combo_kmeans_cluster.addItem(
                    f"Cluster {c} — dens. {dens_media:.0f} ({n_voxels} vx)", userData=c
                )

            self.plotter.add_text("K-Means concluído.", name="status", color="white")
            QMessageBox.information(self, "Concluído", f"K-Means executado ({k} clusters).")

        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha no K-Means:\n{str(e)}")

    def isolate_kmeans_cluster(self):
        if self.kmeans_labels is None:
            QMessageBox.warning(self, "Erro", "Execute o K-Means primeiro.")
            return
        if self.combo_kmeans_cluster.currentIndex() < 0:
            QMessageBox.warning(self, "Erro", "Selecione um cluster.")
            return

        cluster_id = self.combo_kmeans_cluster.currentData()
        denoise_level = self.slider_denoise.value()
        mask = self.kmeans_labels == cluster_id

        try:
            # Carcaça
            self._render_mask_as_mesh(
                mask, denoise_level, f"K-Means — Cluster {cluster_id}",
                context_mask=(self.kmeans_labels >= 0), context_denoise=1
            )
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha na malha:\n{str(e)}")

    def run_watershed(self):
        if self.volume_smooth is None:
            QMessageBox.warning(self, "Erro", "Carregue os dados primeiro!")
            return

        thresh_min = self.slider_thresh_min.value()
        thresh_max = self.slider_thresh_max.value()
        min_dist = self.slider_ws_mindist.value()

        if thresh_min >= thresh_max:
            QMessageBox.warning(self, "Erro", "A densidade mínima deve ser menor que a máxima.")
            return

        self.plotter.add_text("Calculando Watershed...", name="status", color="white")
        self.plotter.update()
        QApplication.processEvents()

        try:
            mask = (self.volume_smooth > thresh_min) & (self.volume_smooth < thresh_max)
            struct_leve = generate_binary_structure(3, 1)
            mask = binary_opening(mask, structure=struct_leve, iterations=1)
            mask = binary_closing(mask, structure=struct_leve, iterations=1)

            if not mask.any():
                QMessageBox.warning(self, "Aviso", "Nenhum voxel na faixa.")
                return

            distancia = distance_transform_edt(mask)
            coords_picos = peak_local_max(distancia, min_distance=min_dist, labels=mask)

            if coords_picos.shape[0] == 0:
                QMessageBox.warning(self, "Aviso", "Nenhum marcador encontrado.")
                return

            marcadores = np.zeros(mask.shape, dtype=bool)
            marcadores[tuple(coords_picos.T)] = True
            marcadores_rotulados, n_marcadores = label(marcadores)

            usar_gradiente = self.combo_ws_relevo.currentIndex() == 0
            relevo = self._gradient_magnitude(self.volume_smooth) if usar_gradiente else -distancia

            regioes = segmentation.watershed(relevo, marcadores_rotulados, mask=mask)
            self.watershed_labels = regioes

            tamanhos = np.bincount(regioes.ravel())
            tamanhos[0] = 0
            ordem_tamanho = np.argsort(tamanhos)[::-1]
            regioes_validas = [int(r) for r in ordem_tamanho if tamanhos[r] > 0][:30]

            self.combo_ws_region.clear()
            for r in regioes_validas:
                self.combo_ws_region.addItem(f"Região {r} ({tamanhos[r]} vx)", userData=r)

            self.plotter.add_text(f"Watershed separou {n_marcadores} regiões.", name="status", color="white")
            QMessageBox.information(self, "Concluído", f"Watershed finalizado com {n_marcadores} regiões.")

        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha no Watershed:\n{str(e)}")

    def isolate_watershed_region(self):
        if self.watershed_labels is None:
            QMessageBox.warning(self, "Erro", "Execute o Watershed primeiro.")
            return
        if self.combo_ws_region.currentIndex() < 0:
            QMessageBox.warning(self, "Erro", "Nenhuma região selecionada.")
            return

        region_id = self.combo_ws_region.currentData()
        denoise_level = self.slider_denoise.value()
        mask = self.watershed_labels == region_id

        try:
            # Carcaça
            self._render_mask_as_mesh(
                mask, denoise_level, f"Watershed — Região {region_id}",
                context_mask=(self.watershed_labels > 0), context_denoise=0
            )
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha na malha:\n{str(e)}")

    def view_all_watershed_regions(self):
        if self.watershed_labels is None:
            QMessageBox.warning(self, "Erro", "Execute o Watershed primeiro.")
            return

        mask = self.watershed_labels > 0
        try:
            self._render_mask_as_mesh(
                mask, 0, "Watershed — Todas as Regiões",
                color_label_volume=self.watershed_labels
            )
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha na malha:\n{str(e)}")

    def clear_scene(self):
        self.plotter.clear()
        self.plotter.add_axes()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AedesAtlasApp()
    window.show()
    sys.exit(app.exec())