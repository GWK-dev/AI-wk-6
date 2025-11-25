# Edge AI Prototype: Recyclable Items Classification
import tensorflow as tf
import numpy as np
import time

class EdgeAIPrototype:
    def __init__(self):
        self.model = None
        self.tflite_model = None
        
    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(6, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model
    
    def generate_data(self):
        X_train = np.random.rand(1000, 128, 128, 3).astype(np.float32)
        y_train = np.random.randint(0, 6, 1000)
        X_test = np.random.rand(200, 128, 128, 3).astype(np.float32)
        y_test = np.random.randint(0, 6, 200)
        return (X_train, y_train), (X_test, y_test)
    
    def convert_to_tflite(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.tflite_model = converter.convert()
        with open('recyclable_model.tflite', 'wb') as f:
            f.write(self.tflite_model)
        print("✅ TFLite model saved: recyclable_model.tflite")
    
    def run_demo(self):
        print("🚀 Edge AI Prototype: Recyclable Classification")
        self.create_model()
        (X_train, y_train), (X_test, y_test) = self.generate_data()
        self.model.fit(X_train, y_train, epochs=5, verbose=0)
        accuracy = self.model.evaluate(X_test, y_test, verbose=0)[1]
        self.convert_to_tflite()
        print(f"📊 Model Accuracy: {accuracy:.4f}")
        print("🎯 Edge AI Benefits: Low latency, enhanced privacy, offline operation")

if __name__ == "__main__":
    demo = EdgeAIPrototype()
    demo.run_demo()