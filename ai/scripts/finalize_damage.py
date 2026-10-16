"""So sánh hai checkpoint bằng val, khóa lựa chọn rồi đánh giá test đúng model đã chọn."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import AI_ROOT


def checkpoint_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(model, split, output, device, imgsz):
    saved = output / 'metrics.json'
    if saved.is_file():
        report = json.loads(saved.read_text())
        if (report['checkpoint_sha256'] != checkpoint_hash(model)
                or report['split'] != split or report['imgsz'] != imgsz):
            raise ValueError(f'Stale report: {saved}; choose a fresh report directory')
        return report
    subprocess.run([sys.executable, str(AI_ROOT / 'scripts/evaluate_damage.py'),
                    '--model', str(model), '--split', split, '--output', str(output),
                    '--device', device, '--imgsz', str(imgsz)], check=True)
    return json.loads(saved.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--external', type=Path, help='Ảnh thực tế ngoài CarDD, nếu đã có')
    a = p.parse_args()
    root = AI_ROOT / 'reports/metrics'
    baseline = evaluate(a.baseline, 'val', root / 'baseline_val', a.device, a.imgsz)
    candidate = evaluate(a.candidate, 'val', root / 'finetune_val', a.device, a.imgsz)
    # Chỉ dùng mask mAP50-95 trên val. Nếu bằng nhau, giữ baseline.
    chosen = candidate if candidate['mask_mAP50_95'] > baseline['mask_mAP50_95'] else baseline
    selected = Path(chosen['checkpoint'])
    manifest = {'selection_metric': 'validation mask mAP50-95',
                'selected_checkpoint': str(selected), 'checkpoint_sha256': checkpoint_hash(selected),
                'imgsz': a.imgsz, 'baseline_val': baseline, 'candidate_val': candidate,
                'test_used_for_selection': False, 'status': 'test_pending'}
    selection_path = AI_ROOT / 'reports/model_selection.json'
    selection_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    test = evaluate(selected, 'test', root / 'selected_test', a.device, a.imgsz)
    sample = next(iter(sorted((AI_ROOT / 'data/yolo/images/val').glob('*.jpg'))))
    subprocess.run([sys.executable, str(AI_ROOT / 'inference/infer.py'), '--model', str(selected),
                    '--source', str(sample), '--output', str(AI_ROOT / 'reports/inference_selected.json'),
                    '--device', a.device, '--imgsz', str(a.imgsz)], check=True)
    external_status = 'pending: no external images provided; CarDD examples are not external validation'
    if a.external:
        subprocess.run([sys.executable, str(AI_ROOT / 'inference/infer.py'), '--model', str(selected),
                        '--source', str(a.external), '--output', str(AI_ROOT / 'reports/inference_external.json'),
                        '--device', a.device, '--imgsz', str(a.imgsz)], check=True)
        external_status = 'predictions generated; no accuracy claim without ground-truth labels'
    manifest.update(status='evaluated', test=test, external_validation=external_status)
    selection_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    lines = ['# Kết quả training và đánh giá', '',
             'Model được chọn theo mask mAP50-95 trên validation, trước khi đánh giá test.', '',
             '| Checkpoint / split | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |',
             '|---|---:|---:|---:|---:|']
    for label, report in [('Baseline / val', baseline), ('Fine-tune / val', candidate), ('Selected / test', test)]:
        lines.append('| ' + label + ' | ' + ' | '.join(f'{report[k]:.4f}' for k in
                     ['box_mAP50','box_mAP50_95','mask_mAP50','mask_mAP50_95']) + ' |')
    lines += ['', f'Checkpoint đã chọn: `{selected}`', '',
              'AP theo 6 lớp: xem `metrics/*/per_class.csv` và `metrics/*/metrics.json`.', '',
              f'Ảnh ngoài CarDD: {external_status}', '',
              'Kết quả này chưa chứng minh độ chính xác trên mọi ảnh thực tế.']
    (AI_ROOT / 'reports/KET_QUA.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Selection and test report saved: {selection_path}')


if __name__ == '__main__':
    main()
