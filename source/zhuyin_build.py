"""產生注音字典：把原始檔裡出現的中文逐字標上注音（依前後文判斷破音字），給畫面自動加注音用。

需要：pip install pypinyin opencc-python-reimplemented（MIT／Apache 授權，只在打包時使用）
輸出格式：{"d": {字: 注音}, "c": {前後文: 注音}}，前後文鍵為「前字+字+後字」「前字+字+_」「_+字+後字」
"""
import re
from collections import Counter, defaultdict
from opencc import OpenCC
from pypinyin import pinyin, Style

CJK = re.compile(r'[㐀-鿿]+')
T2S = OpenCC('t2s')

# 課本標原調；台灣讀音；遊戲裡常用、字典常判斷錯的讀法
FORCE = {'一': 'ㄧ', '不': 'ㄅㄨˋ', '企': 'ㄑㄧˋ', '微': 'ㄨㄟ', '蝸': 'ㄨㄛ', '期': 'ㄑㄧˊ', '垃': 'ㄌㄜˋ', '圾': 'ㄙㄜˋ',
         '危': 'ㄨㄟˊ', '液': 'ㄧˋ', '骰': 'ㄕㄞˇ', '喔': 'ㄛ'}
PHRASE = {
    '數一數': ['ㄕㄨˇ', None, 'ㄕㄨˇ'], '數數': ['ㄕㄨˇ', 'ㄕㄨˇ'], '一起數': [None, None, 'ㄕㄨˇ'], '只數': [None, 'ㄕㄨˇ'],
    '數到': ['ㄕㄨˇ', None], '數完': ['ㄕㄨˇ', None], '數出': ['ㄕㄨˇ', None], '下數': [None, 'ㄕㄨˇ'], '數看': ['ㄕㄨˇ', None],
    '慢慢地': [None, None, '˙ㄉㄜ'], '快快地': [None, None, '˙ㄉㄜ'], '輕輕地': [None, None, '˙ㄉㄜ'], '用力地': [None, None, '˙ㄉㄜ'], '好好地': [None, None, '˙ㄉㄜ'],
    '差一點': ['ㄔㄚ', None, None], '轉圈': ['ㄓㄨㄢˋ', None], '轉一圈': ['ㄓㄨㄢˋ', None, None], '原地': [None, 'ㄉㄧˋ'],
}


def fmt(r):
    # 輕聲符號放在最前面（課本寫法）
    return '˙' + r[:-1] if r.endswith('˙') else r


def read_run(run):
    s = T2S.convert(run)
    if len(s) != len(run):
        s = run
    out = [fmt(x[0]) for x in pinyin(s, style=Style.BOPOMOFO)]
    out = [FORCE.get(ch, r) for ch, r in zip(run, out)]
    for ph, rs in PHRASE.items():
        for m in re.finditer(re.escape(ph), run):
            for k, r in enumerate(rs):
                if r:
                    out[m.start() + k] = r
    return out


def build(text):
    runs = CJK.findall(text)
    per = defaultdict(Counter)
    ctx = defaultdict(Counter)
    for run in runs:
        rs = read_run(run)
        for i, (ch, r) in enumerate(zip(run, rs)):
            if not r or not re.match(r'^[˙ㄅ-ㄩˊˇˋ]+$', r):
                continue
            per[ch][r] += 1
            p, n = run[i - 1] if i else '_', run[i + 1] if i + 1 < len(run) else '_'
            ctx[p + ch + n][r] += 1
            ctx[p + ch + '_'][r] += 1
            ctx['_' + ch + n][r] += 1
    d = {ch: c.most_common(1)[0][0] for ch, c in per.items()}
    c = {}
    for k, cnt in ctx.items():
        r = cnt.most_common(1)[0][0]
        if r != d[k[1]]:
            c[k] = r
    return {'d': d, 'c': c}


if __name__ == '__main__':
    import json, sys
    z = build(open(sys.argv[1], encoding='utf-8').read())
    print(len(z['d']), 'chars,', len(z['c']), 'context rules')
    for t in sys.argv[2:]:
        print(t, ' '.join(z['c'].get((t[i - 1] if i else '_') + ch + (t[i + 1] if i + 1 < len(t) else '_')) or z['c'].get((t[i - 1] if i else '_') + ch + '_') or z['c'].get('_' + ch + (t[i + 1] if i + 1 < len(t) else '_')) or z['d'].get(ch, '?') for i, ch in enumerate(t)))
