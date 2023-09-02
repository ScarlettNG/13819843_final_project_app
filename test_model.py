import unittest
import pandas as pd
from joblib import load


class TestHeartDiseaseModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the model and scaler from disk
        cls.model = load('D:/IP/Birkbeck/MASTER DEGREE/FINAL THESIS/Heart Disease App Code/heart_pred_model.joblib')
        cls.scaler = load('D:/IP/Birkbeck/MASTER DEGREE/FINAL THESIS/Heart Disease App Code/std_scaler.joblib')

    def preprocess_input(self, input_data):
        # Convert the input data to a DataFrame
        df_input = pd.DataFrame([input_data],
                                columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng',
                                         'oldpeak', 'slp', 'caa', 'thall'])

        # Scale the input data
        scaled_input = self.scaler.transform(df_input)

        return scaled_input

    def test_heart_disease(self):
        input_data = (52, 1, 2, 172, 199, 1, 1, 162, 0, 0.5, 2, 0, 3)
        processed_input = self.preprocess_input(input_data)

        prediction = self.model.predict(processed_input)

        self.assertEqual(prediction[0], 1)

    def test_no_heart_disease(self):
        input_data = (67, 1, 0, 160, 286, 0, 0, 108, 1, 1.5, 1, 3, 2)
        processed_input = self.preprocess_input(input_data)

        prediction = self.model.predict(processed_input)

        self.assertEqual(prediction[0], 0)


if __name__ == '__main__':
    unittest.main()
