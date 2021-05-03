import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QLineEdit
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QSize    
import cv2
from gui_utils import CnnGru, read_video, load_model, get_emotion

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(500, 300))    
        self.setWindowTitle("Emotion prediction") 

        self.nameLabel = QLabel(self)
        self.nameLabel.setText('Video path: ')
        self.line = QLineEdit(self)

        self.line.move(80, 20)
        self.line.resize(200, 32)
        self.nameLabel.move(20, 20)

        pybutton = QPushButton('predict', self)
        pybutton.clicked.connect(self.clickMethod)
        pybutton.resize(200,32)
        pybutton.move(80, 60)  

    def pred_emotion(self, vid_path):
        frames = read_video(vid_path)
        input_model = CnnGru(64, 10, 32, 8)
        input_model2 = CnnGru(64, 10, 32, 8)
        model_a = load_model(input_model, './saved_weights/arousal.pth').to('cuda')
        model_v = load_model(input_model2, './saved_weights/valence.pth').to('cuda')        
        arousal = model_a(frames.to('cuda')).squeeze().tolist()
        valence = model_v(frames.to('cuda')).squeeze().tolist()
        print(arousal)
        print(valence)
        emotions = []
        for i in range(len(arousal)):
            a = arousal[i]
            v = valence[i]
            emotion = get_emotion(a,v)
            emotions.append(emotion)
        return emotions

    def display_vid(self,vid_path, emotions):
        cap = cv2.VideoCapture(vid_path)

        width  = cap.get(3)  # float width
        height = cap.get(4)  # float height
        frame_lst = [ i-1 for i in range(40, 401) if i%40 ==0]
        last_emotion = "None"

        i = 0
        frame_count = 0
        while cap.isOpened():
            # Read video capture
            ret, frame = cap.read()
            # # Display each frame
            font = cv2.FONT_HERSHEY_SIMPLEX



            if i in frame_lst:
                last_emotion = emotions[frame_count]
                frame_count +=1


                

            cv2.putText(frame, f'frame:{i}', (0, 100), font, 1, (0,0,255), 1)
            cv2.putText(frame, f'emotion: {last_emotion}', (0, 150), font, 1, (0,0,255), 1)
            cv2.imshow("Emotion prediction", frame)
            i +=1

            # show one frame at a time
            key = cv2.waitKey(0)
            while key not in [ord('q'), ord('k')]:
                key = cv2.waitKey(0)
            # Quit when 'q' is pressed
            if key == ord('q'):
                
                break               
    def clickMethod(self):
        print('file path: ' + self.line.text())
        vid_path = self.line.text()
        emotions = self.pred_emotion(vid_path)

        print(emotions)
        self.display_vid(vid_path,emotions)
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit( app.exec_() )



