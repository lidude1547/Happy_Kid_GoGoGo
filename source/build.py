"""把 app.src.html 和 sounds/ 裡的 mp3、wav 打包成單一檔案 index.html（可離線、可當附件傳送）。

用法：python build.py
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).parent
src = (ROOT / 'app.src.html').read_text(encoding='utf-8')

sounds = {}
MIME = {'.mp3': 'audio/mpeg', '.wav': 'audio/wav'}
for f in sorted((ROOT / 'sounds').iterdir()):
    if f.suffix in MIME:
        sounds[f.stem] = f'data:{MIME[f.suffix]};base64,' + base64.b64encode(f.read_bytes()).decode('ascii')

import sys
sys.path.insert(0, str(ROOT))
from zhuyin_build import build as zhuyin_dict  # 注音字典（需要 pypinyin、opencc）
moe = json.loads((ROOT / 'zhuyin_moe.json').read_text(encoding='utf-8'))  # 從教育部辭典裁切出的詞
src = src.replace('{/*@ZHUYIN@*/}', json.dumps(zhuyin_dict(src, moe), ensure_ascii=False))
marker = '{/*@SOUNDS@*/}'
assert marker in src, 'app.src.html 裡找不到音效標記'
out = src.replace(marker, json.dumps(sounds))
(ROOT / 'index.html').write_text(out, encoding='utf-8')
print(f'index.html：{len(out.encode("utf-8")) // 1024} KB，內含 {len(sounds)} 個音效')
