import unittest
from pathlib import Path
from fastapi.testclient import TestClient
import cv2
import numpy as np

from ai.inference.api import create_app


class StubPredictor:
    model_path, imgsz, confidence = 'test-checkpoint.pt', 640, .25

    def predict(self, source):
        image = cv2.imread(source)
        if image is None:
            raise ValueError('invalid image')
        return {'image': source, 'width': image.shape[1], 'height': image.shape[0],
                'detections': [], 'excluded_tire_flat': 0}


class APITests(unittest.TestCase):
    def test_upload_and_bad_inputs(self):
        with TestClient(create_app(StubPredictor())) as client:
            self.assertEqual(client.get('/health').json()['status'], 'ready')
            ok, buffer = cv2.imencode('.png', np.zeros((20,30,3),dtype=np.uint8))
            self.assertTrue(ok)
            response = client.post('/predict', files={'file': ('../car.png', buffer.tobytes(), 'image/png')})
            self.assertEqual(response.status_code,200)
            result = response.json()['images'][0]
            self.assertEqual(result['image'],'car.png')
            self.assertEqual((result['width'],result['height']), (30,20))
            self.assertEqual(client.post('/predict', files={'file': ('bad.jpg',b'bad','image/jpeg')}).status_code,422)
            self.assertEqual(client.post('/predict', files={'file': ('empty.jpg',b'','image/jpeg')}).status_code,422)
            self.assertEqual(client.post('/predict').status_code,422)


if __name__ == '__main__':
    unittest.main()
