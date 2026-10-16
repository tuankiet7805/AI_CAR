"""Hồi quy mapping, lọc tire flat và định dạng JSON; không cần nạp weights."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import cv2
import numpy as np

from ai.inference.infer import DamagePredictor, export_predictions, map_damage_type
from ai.runtime import CLASS_NAMES, validate_model


class InferenceTests(unittest.TestCase):
    def test_all_class_aliases(self):
        for name, expected in zip(CLASS_NAMES, ['DENT', 'SCRATCH', 'CRACK', 'CRACK', 'CRACK', None]):
            for alias in [name, name.replace(' ', '_'), name.upper()]:
                actual = map_damage_type(alias)
                self.assertEqual(actual.value if actual is not None else None, expected)
        with self.assertRaises(ValueError):
            map_damage_type('unknown')

    def test_filter_preserves_metadata_and_input(self):
        items = [{'class_name': 'tire_flat'}, {'class_name': 'glass_shatter', 'bbox': [1,2,3,4]}]
        before = json.dumps(items)
        out = json.loads(export_predictions(items))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]['damage_type'], 'CRACK')
        self.assertEqual(out[0]['bbox'], [1,2,3,4])
        self.assertEqual(json.dumps(items), before)
        self.assertEqual(json.loads(export_predictions([])), [])

    def test_wrong_checkpoint_rejected(self):
        with self.assertRaises(ValueError):
            validate_model(SimpleNamespace(task='detect', names={}))
        with self.assertRaises(ValueError):
            validate_model(SimpleNamespace(task='segment', names={0:'person'}))
        validate_model(SimpleNamespace(task='segment', names=dict(enumerate(CLASS_NAMES))))

    def test_prediction_payload_and_empty_results(self):
        with TemporaryDirectory() as tmp:
            image = Path(tmp) / 'input.jpg'
            cv2.imwrite(str(image), np.zeros((20,30,3), dtype=np.uint8))
            boxes = [SimpleNamespace(cls=np.array(i), conf=np.array(.8),
                                     xyxy=np.array([[1.,2.,10.,15.]])) for i in range(6)]
            result = SimpleNamespace(boxes=boxes, masks=SimpleNamespace(
                xy=[np.array([[1.,2.],[10.,2.],[10.,15.]]) for _ in boxes]),
                names={i:n.replace(' ','_') for i,n in enumerate(CLASS_NAMES)})
            predictor = DamagePredictor.__new__(DamagePredictor)
            predictor.model = SimpleNamespace(predict=lambda *args, **kwargs: [result])
            predictor.device, predictor.imgsz, predictor.confidence = 'cpu', 640, .25
            out = predictor.predict(str(image))
            self.assertEqual((out['width'],out['height']), (30,20))
            self.assertEqual(out['excluded_tire_flat'],1)
            self.assertEqual(len(out['detections']),5)
            self.assertEqual(out['detections'][3]['damage_type'],'CRACK')
            self.assertEqual(out['detections'][0]['bbox_xyxy'],[1.,2.,10.,15.])
            json.dumps(out, allow_nan=False)
            result.boxes = None
            self.assertEqual(predictor.predict(str(image))['detections'], [])
            image.write_bytes(b'not an image')
            with self.assertRaises(ValueError):
                predictor.predict(str(image))


if __name__ == '__main__':
    unittest.main()
