"""API local nhận multipart ảnh và trả JSON; chạy một worker cho một model."""
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from fastapi import FastAPI, HTTPException, UploadFile

from .infer import DamagePredictor
from ..runtime import AI_ROOT

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def create_app(predictor=None):
    @asynccontextmanager
    async def lifespan(app):
        selected = os.environ.get('CARDD_MODEL')
        imgsz = int(os.environ.get('CARDD_IMGSZ', '640'))
        if predictor is None and not selected:
            manifest = AI_ROOT / 'reports/model_selection.json'
            if manifest.is_file():
                selection = json.loads(manifest.read_text())
                selected = selection['selected_checkpoint']
                if 'CARDD_IMGSZ' not in os.environ:
                    imgsz = selection['imgsz']
        if predictor is None and not selected:
            raise RuntimeError('Set CARDD_MODEL to a trained CarDD checkpoint')
        app.state.predictor = predictor or DamagePredictor(
            selected, device=os.environ.get('CARDD_DEVICE', 'auto'), imgsz=imgsz)
        app.state.model_lock = Lock()
        yield

    app = FastAPI(title='AutoClaimAI Damage Inference', version='1.0', lifespan=lifespan)

    @app.get('/health')
    def health():
        return {'status': 'ready', 'model': app.state.predictor.model_path}

    @app.post('/predict')
    def predict(file: UploadFile):
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if not content:
            raise HTTPException(422, 'Empty image')
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, 'Image exceeds 20 MiB')
        # File tạm riêng cho request; không ghi tên file client vào đường dẫn server.
        with TemporaryDirectory(prefix='cardd-upload-') as tmp:
            path = Path(tmp) / 'image.jpg'
            path.write_bytes(content)
            try:
                with app.state.model_lock:
                    result = app.state.predictor.predict(str(path))
            except ValueError:
                raise HTTPException(422, 'Image cannot be decoded') from None
        result['image'] = Path(file.filename or 'image').name
        predictor_ = app.state.predictor
        return {'schema_version': '1.0', 'model': predictor_.model_path,
                'confidence_threshold': predictor_.confidence, 'imgsz': predictor_.imgsz,
                'coordinate_system': 'pixels; bbox_xyxy=[left,top,right,bottom]',
                'images': [result]}

    return app


app = create_app()
