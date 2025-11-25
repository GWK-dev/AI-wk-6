# Smart Agriculture AI-IoT System
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

class SmartAgricultureAI:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.sensors = [
            "Soil Moisture Sensor", "Temperature Sensor", "Humidity Sensor",
            "NPK Sensor", "pH Sensor", "Solar Radiation Sensor"
        ]
    
    def generate_sensor_data(self):
        data = []
        for day in range(365):
            record = {
                'day': day,
                'soil_moisture': np.random.uniform(10, 60),
                'temperature': np.random.uniform(5, 35),
                'humidity': np.random.uniform(20, 90),
                'nitrogen': np.random.uniform(10, 40),
                'phosphorus': np.random.uniform(5, 25),
                'solar_radiation': np.random.uniform(0, 800)
            }
            record['crop_yield'] = self.calculate_yield(record)
            data.append(record)
        return pd.DataFrame(data)
    
    def calculate_yield(self, conditions):
        base_yield = 5000
        factors = [
            max(0, 1 - abs(conditions['soil_moisture'] - 35) / 35),
            max(0, 1 - abs(conditions['temperature'] - 25) / 25),
            conditions['nitrogen'] / 25
        ]
        return base_yield * np.prod(factors) * np.random.uniform(0.9, 1.1)
    
    def run_system(self):
        print("🌾 Smart Agriculture AI-IoT System")
        print("📡 Sensors:", ", ".join(self.sensors))
        
        data = self.generate_sensor_data()
        X = data.drop('crop_yield', axis=1)
        y = data['crop_yield']
        
        train_size = int(0.8 * len(data))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        
        print(f"📊 Yield Prediction MAE: {mae:.2f} kg/hectare")
        print("🔄 Data Flow: Sensors → Edge Processing → AI Cloud → Farm Actions")
        print("🎯 Benefits: Precision farming, resource optimization, increased yield")

if __name__ == "__main__":
    agri_ai = SmartAgricultureAI()
    agri_ai.run_system()