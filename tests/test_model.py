import unittest
import numpy as np

from src.ml.model import (
    IntentClassifier,
    IntentLabels,
    IntentPrediction,
    TrainingDataGenerator,
)


class TestIntentClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.dataset = TrainingDataGenerator.generate_dataset(samples_per_intent=5)
        cls.classifier = IntentClassifier(model_type='rf')
        cls.metrics = cls.classifier.train(cls.dataset)

    def test_dataset_has_required_columns(self):
        required_cols = {'text', 'intent', 'shape'}
        self.assertTrue(required_cols.issubset(self.dataset.columns))
        self.assertGreater(len(self.dataset), 0)

    def test_training_metrics_are_valid(self):
        self.assertIn('intent_accuracy', self.metrics)
        self.assertIn('shape_accuracy', self.metrics)
        self.assertGreaterEqual(self.metrics['intent_accuracy'], 0.0)
        self.assertLessEqual(self.metrics['intent_accuracy'], 1.0)

    def test_prediction_output_structure(self):
        sample_text = 'calculate area of a circle with radius 5'
        prediction = self.classifier.predict(sample_text)
        self.assertIsInstance(prediction, IntentPrediction)
        self.assertIn(prediction.intent, IntentLabels.INTENTS)
        self.assertGreaterEqual(prediction.confidence, 0.0)
        self.assertLessEqual(prediction.confidence, 1.0)
        self.assertAlmostEqual(sum(prediction.all_probabilities.values()), 1.0, places=3)


if __name__ == '__main__':
    unittest.main()