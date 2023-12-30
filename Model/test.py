import unittest
from model_pipeline import LoadDataset

class TestDataset(unittest.TestCase):
    def test_load_dataset(self):

        # Test loading the dataset
        loader = LoadDataset(train_database_path='./Model/Datasets/lfw_augmented_dataset.db')
        X_train, X_val, X_test, y_train, y_val, y_test, num_classes, df = loader.transform(None)

        # Assert the loaded data
        self.assertEqual(num_classes, 158)

if __name__ == '__main__':
    unittest.main()