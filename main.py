import os
import sys
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from skimage import io, filters, measure
from skimage.filters import threshold_multiotsu
from scipy.ndimage import (gaussian_filter, binary_opening, binary_closing,
                            label, generate_binary_structure)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QSlider, QLabel, QFileDialog,
                             QGroupBox, QComboBox, QMessageBox, QCheckBox, QDialog)
from PyQt6.QtCore import Qt


class AedesAtlasApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Atlas 3D Interativo - Aedes aegypti")
        self.setGeometry(100, 100, 1300, 800)

        self.volume_data = None
        self.volume_smooth = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # PAINEL ESQUERDO: Controles
        control_panel = QWidget()
        control_panel.setFixedWidth(320)
        control_layout = QVBoxLayout(control_panel)

        # 1. Carregamento de Dados
        group_io = QGroupBox("1. Aquisição de Dados")
        io_layout = QVBoxLayout()
        self.btn_load = QPushButton("Carregar Pasta com Fatias (.bmp)")
        self.btn_load.clicked.connect(self.load_directory)
        io_layout.addWidget(self.btn_load)
        group_io.setLayout(io_layout)

        # 2. Visão Global (Volume Completo)
        group_vol = QGroupBox("2. Renderização Volumétrica (Contexto)")
        vol_layout = QVBoxLayout()
        self.btn_render_vol = QPushButton("Exibir Volume Completo (Cores)")
        self.btn_render_vol.clicked.connect(self.show_volume_rendering)
        vol_layout.addWidget(self.btn_render_vol)
        group_vol.setLayout(vol_layout)

        #3. Isolamento de Estruturas
        group_iso = QGroupBox("3. Isolamento de Órgãos (Malha 3D)")
        iso_layout = QVBoxLayout()

        # Ferramenta de calibração: mostra a distribuição real de
        # densidades do volume, com cortes sugeridos por Otsu
        self.btn_histogram = QPushButton("Analisar Densidades (Histograma)")
        self.btn_histogram.clicked.connect(self.show_density_histogram)
        iso_layout.addWidget(self.btn_histogram)

        self.combo_organs = QComboBox()
        self.combo_organs.addItems(["Selecione um Tecido...",
                                    "Exoesqueleto / Músculos Densos",
                                    "Tecidos Moles / Intestino",
                                    "Ajuste Manual"])
        self.combo_organs.currentIndexChanged.connect(self.preset_threshold)

        # Threshold de faixa (min E max)
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

        # Controle novo: intensidade da suavização
        self.lbl_denoise = QLabel("Intensidade de Limpeza (Denoise): 1")
        self.slider_denoise = QSlider(Qt.Orientation.Horizontal)
        self.slider_denoise.setRange(0, 3)
        self.slider_denoise.setValue(1)
        self.slider_denoise.valueChanged.connect(
            lambda v: self.lbl_denoise.setText(f"Intensidade de Limpeza (Denoise): {v}")
        )

        self.btn_extract = QPushButton("Isolar e Gerar Malha 3D")
        self.btn_extract.clicked.connect(self.extract_and_show_mesh)

        iso_layout.addWidget(QLabel("Selecione o alvo anatômico:"))
        iso_layout.addWidget(self.combo_organs)
        iso_layout.addWidget(self.lbl_thresh_min)
        iso_layout.addWidget(self.slider_thresh_min)
        iso_layout.addWidget(self.lbl_thresh_max)
        iso_layout.addWidget(self.slider_thresh_max)
        iso_layout.addWidget(self.lbl_denoise)
        iso_layout.addWidget(self.slider_denoise)
        iso_layout.addWidget(self.btn_extract)
        group_iso.setLayout(iso_layout)

        #4. Controles de Cena (Corte e Limpeza)
        group_scene = QGroupBox("4. Cena Interativa")
        scene_layout = QVBoxLayout()

        self.chk_clip = QCheckBox("Ativar Ferramenta de Corte (Clipping Plane)")
        self.chk_clip.setChecked(False)

        self.btn_clear = QPushButton("Limpar Visualizador")
        self.btn_clear.clicked.connect(self.clear_scene)

        scene_layout.addWidget(self.chk_clip)
        scene_layout.addWidget(self.btn_clear)
        group_scene.setLayout(scene_layout)

        control_layout.addWidget(group_io)
        control_layout.addWidget(group_vol)
        control_layout.addWidget(group_iso)
        control_layout.addWidget(group_scene)
        control_layout.addStretch()

        # PAINEL DIREITO: Visualizador 3D
        self.plotter = QtInteractor(self)
        self.plotter.add_axes()
        self.plotter.add_text("Aguardando importação de dados...", font_size=10, name="status")

        main_layout.addWidget(control_panel)
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

            self.plotter.add_text("Carregando matriz tridimensional (pode levar alguns minutos)...", name="status")
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
                else:
                    print(f"Ignorado: {file}")

            self.volume_data = np.stack(slices, axis=0)

            if self.volume_data.max() <= 1.0:
                self.volume_data = (self.volume_data * 255).astype(np.uint8)

            # Suavização gaussiana
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
                                 "O volume de dados é muito grande para a memória RAM. Aumente a variável 'passo' no código (ex: passo = 3 ou 4).")
        except Exception as e:
            QMessageBox.critical(self, "Erro Crítico", f"Ocorreu um erro ao carregar as imagens:\n{str(e)}")

    def show_volume_rendering(self):
        if self.volume_smooth is None:
            QMessageBox.warning(self, "Erro", "Carregue os dados primeiro!")
            return

        self.plotter.add_text("Calculando Ray Casting Volumétrico...", name="status", color="white")
        self.plotter.update()
        QApplication.processEvents()

        self.clear_scene()
        self.plotter.set_background("black")

        try:
            grid = pv.wrap(self.volume_data)

            if self.chk_clip.isChecked():
                self.plotter.add_volume_clip_plane(
                    grid,
                    cmap="turbo",
                    opacity="linear",
                    shade=False
                )
                self.plotter.add_text("Modo: Renderização Volumétrica (Com Corte)", name="status", color="white")
            else:
                self.plotter.add_volume(
                    grid,
                    cmap="turbo",
                    opacity="linear",
                    shade=False,
                    name="colored_vol"
                )
                self.plotter.add_text("Modo: Renderização Volumétrica Colorida", name="status", color="white")

            self.plotter.reset_camera()

        except Exception as e:
            QMessageBox.critical(self, "Erro na Renderização", f"Falha ao gerar o volume:\n{str(e)}")

    def update_thresh_label(self, _value=None):
        self.lbl_thresh_min.setText(f"Densidade Mínima: {self.slider_thresh_min.value()}")
        self.lbl_thresh_max.setText(f"Densidade Máxima: {self.slider_thresh_max.value()}")

    def preset_threshold(self):
        idx = self.combo_organs.currentIndex()
        if idx == 1:
            # Exoesqueleto / Músculos Densos: só a faixa alta de densidade
            self.slider_thresh_min.setValue(160)
            self.slider_thresh_max.setValue(254)
        elif idx == 2:
            # Tecidos Moles / Intestino: faixa média, excluindo
            # explicitamente a densidade do exoesqueleto (acima de ~150)
            self.slider_thresh_min.setValue(50)
            self.slider_thresh_max.setValue(140)
        self.update_thresh_label()

    def show_density_histogram(self):
        if self.volume_smooth is None:
            QMessageBox.warning(self, "Erro", "Carregue os dados primeiro!")
            return

        # Ignora voxels muito próximos de zero
        dados = self.volume_smooth[self.volume_smooth > 15].ravel()

        if dados.size == 0:
            QMessageBox.warning(self, "Aviso", "Volume parece vazio nessa faixa (só fundo/ar).")
            return

        try:
            cortes = threshold_multiotsu(dados, classes=4)
        except Exception:
            cortes = threshold_multiotsu(dados, classes=3)

        dialog = QDialog(self)
        dialog.setWindowTitle("Distribuição de Densidades do Volume")
        dialog.resize(720, 520)
        layout = QVBoxLayout(dialog)

        fig = Figure(figsize=(7, 4.5))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.hist(dados, bins=150, color="#4fa8ff", log=True)
        for c in cortes:
            ax.axvline(c, color="red", linestyle="--")
            ax.text(c, ax.get_ylim()[1] * 0.85, f"{c:.0f}", color="red", rotation=90)
        ax.set_xlabel("Densidade (0-255)")
        ax.set_ylabel("Nº de voxels (escala log)")
        ax.set_title("Histograma de Densidade (ar/fundo excluído)")
        layout.addWidget(canvas)

        texto_cortes = ", ".join(f"{c:.0f}" for c in cortes)
        lbl = QLabel(
            f"Cortes sugeridos (Otsu): {texto_cortes}\n\n"
            "O Otsu só separa os grupos estatisticamente — ele NÃO sabe qual grupo é "
            "'exoesqueleto' ou 'músculo'. Para descobrir qual faixa é qual estrutura:\n"
            "1. Feche esta janela.\n"
            "2. Selecione 'Ajuste Manual' e use um dos intervalos entre dois cortes "
            "(ex.: entre o 2º e o 3º valor acima) nos sliders Min/Max.\n"
            "3. Gere a malha e veja visualmente qual parte do inseto aparece.\n"
            "4. Repita para cada faixa até identificar todas — aí você pode ajustar os "
            "presets do combo 'Exoesqueleto'/'Tecidos Moles' no código com os valores certos."
        )
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        dialog.exec()

    def extract_and_show_mesh(self):
        if self.volume_smooth is None:
            QMessageBox.warning(self, "Erro", "Carregue os dados primeiro!")
            return

        thresh_min = self.slider_thresh_min.value()
        thresh_max = self.slider_thresh_max.value()
        denoise_level = self.slider_denoise.value()  # 0 a 3 iterações de limpeza

        if thresh_min >= thresh_max:
            QMessageBox.warning(self, "Erro", "Densidade Mínima precisa ser menor que a Máxima.")
            return

        self.plotter.add_text(
            f"Extraindo Órgãos (Densidade {thresh_min}-{thresh_max})...", name="status", color="white"
        )
        self.plotter.update()
        QApplication.processEvents()

        try:
            mask = (self.volume_smooth > thresh_min) & (self.volume_smooth < thresh_max)

            if not mask.any():
                QMessageBox.warning(self, "Aviso", "Nenhum tecido encontrado nessa faixa de densidade.")
                return

            struct_leve = generate_binary_structure(3, 1)
            if denoise_level > 0:
                mask = binary_opening(mask, structure=struct_leve, iterations=1)
                mask = binary_closing(mask, structure=struct_leve, iterations=1)

            struct = generate_binary_structure(3, 2)
            labeled, n_labels = label(mask, structure=struct)
            if n_labels == 0:
                QMessageBox.warning(self, "Aviso", "Nenhum tecido denso encontrado nesse limiar.")
                return

            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0  # ignora o fundo (label 0)

            if sizes.max() == 0:
                QMessageBox.warning(self, "Aviso", "Nenhum tecido denso encontrado nesse limiar.")
                return

            min_voxels = {0: 1, 1: 4, 2: 10, 3: 25}.get(denoise_level, 4)
            labels_validos = np.where(sizes >= min_voxels)[0]
            mask_limpa = np.isin(labeled, labels_validos)

            verts, faces, normals, values = measure.marching_cubes(
                mask_limpa.astype(np.float32), level=0.5
            )
            faces_pv = np.hstack(
                [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
            ).astype(np.int64)
            mesh = pv.PolyData(verts, faces_pv)

            # Suavização Laplaciana
            mesh = mesh.smooth(n_iter=60, relaxation_factor=0.15)
            mesh = mesh.compute_normals(auto_orient_normals=True, inplace=False)

            coords = np.clip(
                np.round(mesh.points).astype(int),
                0,
                np.array(self.volume_data.shape) - 1
            )
            densidade_por_vertice = self.volume_data[
                coords[:, 0], coords[:, 1], coords[:, 2]
            ]
            mesh["Densidade"] = densidade_por_vertice

            self.clear_scene()
            self.plotter.set_background("black")

            if self.chk_clip.isChecked():
                self.plotter.add_mesh_clip_plane(
                    mesh,
                    scalars="Densidade",
                    cmap="turbo",
                    clim=[0, 255],
                    smooth_shading=True,
                    specular=0.5
                )
                self.plotter.add_text(f"Modo: Órgãos Internos com Corte ({thresh_min}-{thresh_max})", name="status", color="white")
            else:
                self.plotter.add_mesh(
                    mesh,
                    scalars="Densidade",
                    cmap="turbo",
                    clim=[0, 255],
                    smooth_shading=True,
                    specular=0.5,
                    name="iso_mesh"
                )
                self.plotter.add_text(f"Modo: Órgãos Internos Coloridos ({thresh_min}-{thresh_max})", name="status", color="white")

            self.plotter.reset_camera()

        except Exception as e:
            QMessageBox.critical(self, "Erro na Extração", f"Não foi possível gerar a malha:\n{str(e)}")

    def clear_scene(self):
        self.plotter.clear()
        self.plotter.add_axes()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AedesAtlasApp()
    window.show()
    sys.exit(app.exec())