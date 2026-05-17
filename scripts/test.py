# coding:utf-8
"""测试脚本：批量识别验证码，生成 HTML 测试报告"""
import base64
import json
import os
import platform
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from app import recognize


def _get_env_info(model_path):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        device = ', '.join(g.name for g in gpus)
    else:
        device = 'CPU'

    return {
        'python': platform.python_version(),
        'tensorflow': tf.__version__,
        'device': device,
        'model_path': os.path.basename(model_path),
    }


def _image_to_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


def main():
    test_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/test/'
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'saved_model/model.weights.h5'

    files = sorted(f for f in os.listdir(test_dir) if f.endswith('.png'))
    if not files:
        print(f'未找到测试图片: {test_dir}')
        return

    print(f'测试图片: {len(files)} 张')
    print(f'模型: {os.path.basename(model_path)}')
    print()

    errors = []
    correct = 0
    total_time = 0.0

    for i, f in enumerate(files, 1):
        label = f.split('.')[0]
        path = os.path.join(test_dir, f)

        t0 = time.time()
        pred = recognize(path, model_path=model_path)
        dt = time.time() - t0
        total_time += dt

        if pred == label:
            correct += 1
        else:
            errors.append({
                'file': f,
                'label': label,
                'pred': pred,
                'image_b64': _image_to_b64(path),
            })
            print(f'  [{i}/{len(files)}] {label} -> {pred}')

        if i % 50 == 0:
            print(f'  进度: {i}/{len(files)}  准确率: {correct / i:.1%}')

    total = len(files)
    accuracy = correct / total * 100 if total else 0
    avg_ms = total_time / total * 1000 if total else 0

    print(f'\n结果: {correct}/{total}  准确率: {accuracy:.1f}%  平均耗时: {avg_ms:.1f}ms')

    # 生成报告
    report_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total': total,
        'correct': correct,
        'accuracy': round(accuracy, 1),
        'avg_time_ms': round(avg_ms, 1),
        'environment': _get_env_info(model_path),
        'errors': errors,
    }

    template_path = os.path.join(os.path.dirname(__file__), 'report_template.html')
    with open(template_path, 'r') as f:
        html = f.read()

    html = html.replace('__REPORT_DATA__', json.dumps(report_data, ensure_ascii=False))

    os.makedirs('output', exist_ok=True)
    report_path = 'output/test_report.html'
    with open(report_path, 'w') as f:
        f.write(html)

    print(f'报告已生成: {report_path}')


if __name__ == '__main__':
    main()
