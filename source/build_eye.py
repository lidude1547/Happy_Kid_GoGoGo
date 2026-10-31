"""把 app.src.html 加上眼睛控制層（eye/），打包成單一檔案 eye.html（不用手、用鏡頭眼神操作的版本）。

用法：python build_eye.py
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).parent
src = (ROOT / 'app.src.html').read_text(encoding='utf-8')
css = (ROOT / 'eye' / 'eye.css').read_text(encoding='utf-8')
html = (ROOT / 'eye' / 'eye.html').read_text(encoding='utf-8')
js = (ROOT / 'eye' / 'eye.js').read_text(encoding='utf-8')


def rep(s, a, b, n=1):
    assert s.count(a) == n, f'找不到：{a[:60]}（{s.count(a)} 次）'
    return s.replace(a, b)


s = src
# 和一般版分開儲存，互不影響
s = rep(s, "const KEY = 'doudou-v1';", "const KEY = 'doudou-eye-v1';")
s = rep(s, "const SEEN_KEY = 'doudou-seen';", "const SEEN_KEY = 'doudou-eye-seen';")
s = s.replace("'doudou-art-'", "'doudou-eye-art-'")
s = rep(s, '<title>豆豆遊樂園</title>', '<title>豆豆眼睛遊樂園</title>')
s = rep(s, '<link rel="manifest" href="manifest.webmanifest">\n', '')
s = rep(s, "if ('serviceWorker' in navigator && /^https?:$/.test(location.protocol)) navigator.serviceWorker.register('sw.js').catch(() => {});\n", '')
# 眼睛控制層
s = rep(s, '</style>', css + '</style>')
s = rep(s, '<script>\nconst PRISTINE', html + '\n<script>\nconst PRISTINE')
s = rep(s, '/* ================= 啟動 ================= */', js + '\n/* ================= 啟動 ================= */')
s = rep(s, 'cur() ? goHome() : renderWho();\n</script>', 'eyeBoot();\n</script>')

sounds = {}
MIME = {'.mp3': 'audio/mpeg', '.wav': 'audio/wav'}
for f in sorted((ROOT / 'sounds').iterdir()):
    if f.suffix in MIME:
        sounds[f.stem] = f'data:{MIME[f.suffix]};base64,' + base64.b64encode(f.read_bytes()).decode('ascii')
s = rep(s, '{/*@SOUNDS@*/}', json.dumps(sounds))
(ROOT / 'eye.html').write_text(s, encoding='utf-8')
print(f'eye.html：{len(s.encode("utf-8")) // 1024} KB')
