import unittest
import numpy as np
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from model_pipeline import LoadDataset, Preprocess, TrainModel, VisualizeFeatureMaps, EvaluateModel
from model_pipeline import PreprocessVGG16, TrainModelVGG16
from model_pipeline import PreprocessEfficientNet, TrainModelEfficientNet
from server.model import predict

# All done by: Dimitrios

class TestDataset(unittest.TestCase):
    def test_load_dataset(self):

        # Test loading the dataset
        loader = LoadDataset(train_database_path='./Model/Datasets/lfw_augmented_dataset.db')
        X_train, X_val, X_test, y_train, y_val, y_test, num_classes, df = loader.transform(None)

        # Assert the loaded data
        self.assertEqual(num_classes, 158)


class TestPipelines(unittest.TestCase):
    
    def test_pipeline_v1(self):

        pipeline_v1 = Pipeline([
            ('load_dataset', LoadDataset(train_database_path='./Model/Datasets/lfw_augmented_dataset.db')),
            ('preprocess', Preprocess()),
            ('train_model', TrainModel()),
            ('visualize_feature_maps', VisualizeFeatureMaps()),
            ('evaluate_model', EvaluateModel())
        ])

        # Run pipeline_v1 with sample data
        accuracy, precision, recall, f1, model = pipeline_v1.fit_transform(None)
        num_of_layers = 11
        # Add assertions based on the expected outcome of the pipeline
        assert accuracy < 1 
        assert precision < 1
        assert recall < 1
        assert f1 < 1
        self.assertIsInstance(model, Sequential)
        self.assertTrue(hasattr(model, 'layers') and len(model.layers) == num_of_layers) 

    def test_pipeline_v2(self):

        pipeline_v2 = Pipeline([
            ('load_dataset', LoadDataset(train_database_path='./Model/Datasets/lfw_augmented_dataset.db')),
            ('preprocess', PreprocessVGG16()),
            ('train_model', TrainModelVGG16()),
            ('evaluate_model', EvaluateModel())
        ])

        # Run pipeline_v2 with sample data
        accuracy, precision, recall, f1, model = pipeline_v2.fit_transform(None)
        num_of_layers = 4
        print(len(model.layers))
        # Add assertions based on the expected outcome of the pipeline
        assert accuracy < 1 
        assert precision < 1
        assert recall < 1
        assert f1 < 1
        self.assertIsInstance(model, Sequential)
        self.assertTrue(hasattr(model, 'layers') and len(model.layers) == num_of_layers) 

    def test_pipeline_v3(self):

        pipeline_v3 = Pipeline([
            ('load_dataset', LoadDataset(train_database_path='./Model/Datasets/lfw_augmented_dataset.db')),
            ('preprocess', PreprocessEfficientNet()),
            ('train_model', TrainModelEfficientNet()),
            ('evaluate_model', EvaluateModel())
        ])

        # Run pipeline_v3 with sample data
        accuracy, precision, recall, f1, model = pipeline_v3.fit_transform(None)
        num_of_layers = 5
        print(len(model.layers))
        # Add assertions based on the expected outcome of the pipeline
        assert accuracy < 1 
        assert precision < 1
        assert recall < 1
        assert f1 < 1
        self.assertIsInstance(model, Sequential)
        self.assertTrue(hasattr(model, 'layers') and len(model.layers) == num_of_layers) 


class TestPrediction(unittest.TestCase):
    def test_predict(self):
        # image that that doesn't match the required size for the model for testing
        dummy_image = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8) 
        predicted_name = predict(dummy_image)
        self.assertIsNotNone(predicted_name)


if __name__ == '__main__':
    unittest.main()