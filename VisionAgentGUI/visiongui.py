import sys
import cv2 as cv
import numpy as np
import winsound
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                             QFileDialog, QComboBox, QVBoxLayout, QHBoxLayout,
                             QWidget)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class VisionAgentGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ビジョンエージェント")
        self.setGeometry(100, 100, 300, 200)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.label = QLabel("ようこそ！", self)
        self.layout.addWidget(self.label)

        self.initial_button_layout = QHBoxLayout()
        self.function_button_layout = QHBoxLayout()
        self.layout.addLayout(self.initial_button_layout)
        self.layout.addLayout(self.function_button_layout)

        self.create_buttons()
        self.show()

    def create_buttons(self):
        # 初期メニューのボタン
        self.orim_button = QPushButton("オリム")
        self.traffic_button = QPushButton("交通弱者ゾーン通知")
        self.panorama_button = QPushButton("パノラマ")
        self.effect_button = QPushButton("特殊効果")
        self.quit_button = QPushButton("終了")

        self.initial_button_layout.addWidget(self.orim_button)
        self.initial_button_layout.addWidget(self.traffic_button)
        self.initial_button_layout.addWidget(self.panorama_button)
        self.initial_button_layout.addWidget(self.effect_button)
        self.initial_button_layout.addWidget(self.quit_button)

        self.orim_button.clicked.connect(self.orim_function)
        self.traffic_button.clicked.connect(self.traffic_function)
        self.panorama_button.clicked.connect(self.panorama_function)
        self.effect_button.clicked.connect(self.effect_function)
        self.quit_button.clicked.connect(self.quit_program)

    def clear_function_button_layout(self):
        # 機能ボタンのレイアウトをクリア
        while self.function_button_layout.count():
            item = self.function_button_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    # --- オリム機能 ---
    def orim_function(self):
        self.label.setText("オリム機能を選択しました")
        self.clear_function_button_layout()
        self.orim_ui()

    def orim_ui(self):
        # 画像オリム用ボタン
        self.file_button = QPushButton("ファイル")
        self.paint_button = QPushButton("ペイント")
        self.cut_button = QPushButton("切り取り")
        self.inc_button = QPushButton("+")
        self.dec_button = QPushButton("-")
        self.save_button = QPushButton("保存")

        self.function_button_layout.addWidget(self.file_button)
        self.function_button_layout.addWidget(self.paint_button)
        self.function_button_layout.addWidget(self.cut_button)
        self.function_button_layout.addWidget(self.inc_button)
        self.function_button_layout.addWidget(self.dec_button)
        self.function_button_layout.addWidget(self.save_button)

        self.file_button.clicked.connect(self.fileOpenFunction)
        self.paint_button.clicked.connect(self.paintfunction)
        self.cut_button.clicked.connect(self.cutfunction)
        self.inc_button.clicked.connect(self.incfunction)
        self.dec_button.clicked.connect(self.decfunction)
        self.save_button.clicked.connect(self.savefunction)

        self.BrushSiz = 5
        self.LColor, self.RColor = (255, 0, 0), (0, 0, 255)

    def fileOpenFunction(self):
        # 画像ファイルを開く
        fname = QFileDialog.getOpenFileName(self, 'ファイルを開く', './')
        self.img = cv.imread(fname[0])
        if self.img is None:
            return
        self.img_show = np.copy(self.img)
        cv.imshow('ペインティング', self.img_show)
        self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)
        self.mask[:, :] = cv.GC_PR_BGD

    def paintfunction(self):
        # マウスイベントで描画設定
        cv.setMouseCallback('ペインティング', self.painting)

    def painting(self, event, x, y, flags, param):
        # 左：前景、右：背景として描画
        if event == cv.EVENT_LBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON):
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.LColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_FGD, -1)
        elif event == cv.EVENT_RBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON):
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.RColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_BGD, -1)
        cv.imshow('ペインティング', self.img_show)

    def cutfunction(self):
        # GrabCutで前景抽出
        background = np.zeros((1, 65), np.float64)
        foreground = np.zeros((1, 65), np.float64)
        cv.grabCut(self.img, self.mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)
        mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
        self.grabImg = self.img * mask2[:, :, np.newaxis]
        cv.imshow('切り取り結果', self.grabImg)

    def incfunction(self):
        self.BrushSiz = min(20, self.BrushSiz + 1)

    def decfunction(self):
        self.BrushSiz = max(1, self.BrushSiz - 1)

    def savefunction(self):
        fname = QFileDialog.getSaveFileName(self, '保存', './')
        if fname[0]:
            cv.imwrite(fname[0], self.grabImg)

    # --- 交通弱者保護ゾーン ---
    def traffic_function(self):
        self.label.setText("交通弱者ゾーン機能を選択しました")
        self.clear_function_button_layout()
        self.traffic_ui()

    def traffic_ui(self):
        self.sign_button = QPushButton("標識")
        self.road_button = QPushButton("道路")
        self.recognition_button = QPushButton("認識")

        self.function_button_layout.addWidget(self.sign_button)
        self.function_button_layout.addWidget(self.road_button)
        self.function_button_layout.addWidget(self.recognition_button)

        self.sign_button.clicked.connect(self.sign_function)
        self.road_button.clicked.connect(self.road_function)
        self.recognition_button.clicked.connect(self.recognition_function)

        self.signFiles = [['child.png','子供'], ['elder.png','高齢者'], ['disabled.png','障がい者']]
        self.signImgs = []

    def sign_function(self):
        self.signImgs.clear()
        for fname, _ in self.signFiles:
            img = cv.imread(fname)
            if img is not None:
                self.signImgs.append(img)
                cv.imshow(fname, img)

    def road_function(self):
        if not self.signImgs:
            self.label.setText('まず標識を読み込んでください。')
            return
        fname, _ = QFileDialog.getOpenFileName(self, '道路画像を開く', '', 'Images (*.png *.jpg *.bmp)')
        if fname:
            self.roadImg = cv.imread(fname)
            if self.roadImg is not None:
                cv.imshow('道路シーン', self.roadImg)

    def recognition_function(self):
        if not hasattr(self, 'roadImg') or self.roadImg is None:
            self.label.setText('まず道路映像を入力してください。')
            return

        # SIFTによる特徴検出とマッチング
        sift = cv.SIFT_create()
        grayRoad = cv.cvtColor(self.roadImg, cv.COLOR_BGR2GRAY)
        road_kp, road_des = sift.detectAndCompute(grayRoad, None)

        best_match_count = 0
        best_match_img = None
        best_match_kp = None
        best_good_matches = None
        best_index = -1

        matcher = cv.BFMatcher()

        for idx, img in enumerate(self.signImgs):
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            matches = matcher.knnMatch(des, road_des, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good_matches) > best_match_count:
                best_match_count = len(good_matches)
                best_match_img = img
                best_match_kp = kp
                best_good_matches = good_matches
                best_index = idx

        if best_match_count >= 4:
            src_pts = np.float32([best_match_kp[m.queryIdx].pt for m in best_good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([road_kp[m.trainIdx].pt for m in best_good_matches]).reshape(-1, 1, 2)
            M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            h, w = best_match_img.shape[:2]
            pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts, M)
            self.roadImg = cv.polylines(self.roadImg, [np.int32(dst)], True, (0,255,0), 3, cv.LINE_AA)

            cv.imshow("標識検出", self.roadImg)
            self.label.setText(self.signFiles[best_index][1] + "保護ゾーンです。30kmで徐行してください。")
            winsound.Beep(3000, 500)
        else:
            self.label.setText("標識が検出されませんでした。")

    # --- パノラマ機能 ---
    def panorama_function(self):
        self.label.setText("パノラマ機能を選択しました")
        self.clear_function_button_layout()
        self.panorama_ui()

    def panorama_ui(self):
        self.load_video_button = QPushButton("ビデオ読み込み", self)
        self.collect_from_video_button = QPushButton("フレーム収集", self)
        self.show_button = QPushButton("フレーム表示", self)
        self.stitch_button = QPushButton("スティッチング", self)
        self.save_pano_button = QPushButton("保存", self)

        self.function_button_layout.addWidget(self.load_video_button)
        self.function_button_layout.addWidget(self.collect_from_video_button)
        self.function_button_layout.addWidget(self.show_button)
        self.function_button_layout.addWidget(self.stitch_button)
        self.function_button_layout.addWidget(self.save_pano_button)

        self.load_video_button.clicked.connect(self.load_video_function)
        self.collect_from_video_button.clicked.connect(self.collect_from_video)
        self.show_button.clicked.connect(self.show_function)
        self.stitch_button.clicked.connect(self.stitch_function)
        self.save_pano_button.clicked.connect(self.save_panorama_function)

        self.collect_from_video_button.setEnabled(False)
        self.show_button.setEnabled(False)
        self.stitch_button.setEnabled(False)
        self.save_pano_button.setEnabled(False)
        self.imgs = []

    def load_video_function(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'ビデオを開く', '', 'Videos (*.mp4 *.avi *.mov)')
        if fname:
            self.cap = cv.VideoCapture(fname)
            self.collect_from_video_button.setEnabled(True)

    def collect_from_video(self):
        # キー操作でフレームを収集
        self.imgs = []
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            cv.imshow("フレーム", frame)
            key = cv.waitKey(30)
            if key == ord('c'):
                self.imgs.append(frame.copy())  # 'c'で保存
            elif key == ord('q'):
                break
        self.cap.release()
        cv.destroyAllWindows()
        self.show_button.setEnabled(True)
        self.stitch_button.setEnabled(True)
        self.save_pano_button.setEnabled(False)

    def show_function(self):
        # 収集したフレーム表示
        for idx, img in enumerate(self.imgs):
            cv.imshow(f"フレーム {idx}", img)
            cv.waitKey(500)
        cv.destroyAllWindows()

    def stitch_function(self):
        # パノラマ合成
        if not self.imgs:
            print("❗ フレームがありません。収集してください。")
            return

        base_shape = (480, 640)
        resized_imgs = [cv.resize(img, base_shape) for img in self.imgs]

        # OpenCVバージョンに応じたステッチャーの作成
        try:
            stitcher = cv.Stitcher_create()
        except AttributeError:
            stitcher = cv.Stitcher.create()

        status, pano = stitcher.stitch(resized_imgs)
        if status == cv.Stitcher_OK:
            self.pano = pano
            cv.imshow("パノラマ結果", self.pano)
            cv.waitKey(1)
            self.save_pano_button.setEnabled(True)
        else:
            print(f"❌ スティッチング失敗: ステータス = {status}")
            self.label.setText("スティッチングに失敗しました。他のフレームで再試行してください。")

    def save_panorama_function(self):
        if hasattr(self, 'pano'):
            fname, _ = QFileDialog.getSaveFileName(self, 'パノラマ保存', '', 'Images (*.png *.jpg *.bmp)')
            if fname:
                cv.imwrite(fname, self.pano)

    # --- 特殊効果機能 ---
    def effect_function(self):
        self.label.setText("特殊効果機能を選択しました")
        self.clear_function_button_layout()
        self.effect_ui()

    def effect_ui(self):
        self.picture_button = QPushButton("画像を開く", self)
        self.save_effect_button = QPushButton("保存", self)
        self.effect_combo = QComboBox(self)
        self.effect_combo.addItems(["エンボス", "カートゥーン", "スケッチ（グレー）", "スケッチ（カラー）", "油絵風"])

        self.function_button_layout.addWidget(self.picture_button)
        self.function_button_layout.addWidget(self.save_effect_button)
        self.function_button_layout.addWidget(self.effect_combo)

        self.picture_button.clicked.connect(self.picture_open_function)
        self.save_effect_button.clicked.connect(self.save_function)
        self.effect_combo.currentIndexChanged.connect(self.apply_selected_effect)
        self.save_effect_button.setEnabled(False)

    def picture_open_function(self):
        fname = QFileDialog.getOpenFileName(self, '画像読み込み', './')
        self.img = cv.imread(fname[0])
        if self.img is None:
            return
        self.apply_selected_effect()
        self.save_effect_button.setEnabled(True)

    def emboss_function(self):
        # エンボス効果
        femboss = np.array([[-1.0, 0.0, 0.0],
                            [ 0.0, 0.0, 0.0],
                            [ 0.0, 0.0, 1.0]])
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        gray16 = np.int16(gray)
        self.emboss = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss) + 128, 0, 255))
        cv.imshow('エンボス', self.emboss)

    def cartoon_function(self):
        # カートゥーン効果
        self.cartoon = cv.stylization(self.img, sigma_s=60, sigma_r=0.45)
        cv.imshow('カートゥーン', self.cartoon)

    def sketch_function(self):
        # スケッチ（グレー・カラー）
        self.sketch_gray, self.sketch_color = cv.pencilSketch(
            self.img, sigma_s=60, sigma_r=0.07, shade_factor=0.02
        )
        cv.imshow('スケッチ（グレー）', self.sketch_gray)
        cv.imshow('スケッチ（カラー）', self.sketch_color)

    def oil_function(self):
        # 油絵風効果
        self.oil = cv.xphoto.oilPainting(self.img, 10, 1, cv.COLOR_BGR2Lab)
        cv.imshow('油絵風', self.oil)

    def apply_selected_effect(self):
        # コンボボックスで選択された効果を適用
        if not hasattr(self, 'img') or self.img is None:
            return
        i = self.effect_combo.currentIndex()
        if i == 0:
            self.emboss_function()
        elif i == 1:
            self.cartoon_function()
        elif i == 2 or i == 3:
            self.sketch_function()
        elif i == 4:
            self.oil_function()

    def save_function(self):
        fname = QFileDialog.getSaveFileName(self, '保存', './')
        if not fname[0]:
            return
        i = self.effect_combo.currentIndex()
        if i == 0:
            cv.imwrite(fname[0], self.emboss)
        elif i == 1:
            cv.imwrite(fname[0], self.cartoon)
        elif i == 2:
            cv.imwrite(fname[0], self.sketch_gray)
        elif i == 3:
            cv.imwrite(fname[0], self.sketch_color)
        elif i == 4:
            cv.imwrite(fname[0], self.oil)

    # --- 終了処理 ---
    def quit_program(self):
        try:
            cv.destroyAllWindows()  # すべてのウィンドウを閉じる
        except Exception as e:
            print("OpenCVウィンドウ終了エラー:", e)
        self.close()
