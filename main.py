import cv2
import kivy
from kivy.app import App
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
import matplotlib.pyplot as plt
from kivy.clock import Clock

import tensorflow as tf

# noinspection PyUnresolvedReferences
from data import get_data_mnist
import numpy as np
import random


class Predict:
    def __init__(self):
        self.interpreter = tf.lite.Interpreter("content/chuviettay/mnist.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.X_test, _ = get_data_mnist()

        self.index: int

    def process_predict(self):
        input_shape = self.input_details[0]['shape']
        input_tensor = np.array(np.expand_dims(self.X_test[self.index], -1), dtype=np.float32)
        input_tensor = np.array(np.expand_dims(input_tensor, 0), dtype=np.float32)
        print(input_tensor.shape)

        # set the tensor to point the input data to be inferrd
        input_index = self.interpreter.get_input_details()[0]["index"]
        self.interpreter.set_tensor(input_index, input_tensor)

        # run the inference
        self.interpreter.invoke()
        Y_pred = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        y_pred = np.argmax(Y_pred, axis=0)

        print(y_pred)
        return y_pred

    def start(self, dt=0):
        self.index = random.randint(0, 10000 - 1)
        
        plt.close()

        plt.imshow(self.X_test[self.index])

        y_pred = self.process_predict()
        plt.title("Result of the predict: " + str(y_pred))


class MyBoxLayout(BoxLayout):
    def __init__(self, **kwargs):
        super(MyBoxLayout, self).__init__(**kwargs)
        imgPre.start()
        # self.__myplot()
        self.screen = FigureCanvasKivyAgg(plt.gcf())
        self.add_widget(self.screen)
        Clock.schedule_interval(self.myplot, 2)

    def myplot(self, dt):
        # file_name = "tmp.png"
        # img = cv2.imread(file_name)
        # plt.imshow(img)
        self.remove_widget(self.screen)
        imgPre.start()
        self.screen = FigureCanvasKivyAgg(plt.gcf())
        self.add_widget(self.screen)

        pass


class MyApp(App):

    def build(self):
        return MyBoxLayout()


if __name__ == "__main__":
    imgPre = Predict()
    MyApp().run()
