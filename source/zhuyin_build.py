"""產生注音字典：以教育部《重編國語辭典修訂本》（萌典資料）為準，替原始檔裡出現的中文標注音。

讀音的決定順序：
  1. 教育部辭典裡有的「詞」（最長的先用），整個詞照辭典的讀音
  2. 單字：pypinyin 依前後文的讀音，如果不在教育部辭典這個字的讀音裡，就改用辭典的第一個讀音
  3. 課本規則與遊戲常用詞（一、不標原調；數一數的「數」念ㄕㄨˇ…），最後覆蓋
輸出：{"d": {字: 注音}, "w": {詞: [每個字的注音]}}，畫面上用「最長詞優先」比對。

打包時只需要 zhuyin_moe.json（從教育部辭典裁切出原始檔用得到的詞）；
原始檔有新的詞時，用 `python zhuyin_build.py --trim app.src.html moe_index.json` 重新裁切。
需要：pip install pypinyin opencc-python-reimplemented jieba（只在打包時使用）
辭典的詞只在和斷詞（jieba）的詞界對齊時才使用，避免「兔兔和熊熊」被誤配成冷僻詞「和熊」。
"""
import json, re, sys
from collections import Counter, defaultdict
from opencc import OpenCC
from pypinyin import pinyin, Style
import jieba, logging
jieba.setLogLevel(logging.WARNING)

CJK = re.compile(r'[㐀-鿿]+')
ZY = re.compile(r'^˙?[ㄅ-ㄩ]+[ˊˇˋ]?$')
T2S = OpenCC('t2s')
MAXW = 6

FORCE = {'一': 'ㄧ', '不': 'ㄅㄨˋ', '骰': 'ㄕㄞˇ'}
# 不在辭典詞條裡時，這些字用孩子課本的常用讀音（逐一對照原始檔裡的用法確認過）
DEFAULT_PREF = {'得': '˙ㄉㄜ', '著': '˙ㄓㄜ', '誰': 'ㄕㄟˊ', '為': 'ㄨㄟˊ', '檔': 'ㄉㄤˇ', '和': 'ㄏㄢˋ', '佛': 'ㄈㄛˊ', '勒': 'ㄌㄜˋ', '切': 'ㄑㄧㄝ', '哇': 'ㄨㄚ'}
PHRASE = {
    '數一數': ['ㄕㄨˇ', 'ㄧ', 'ㄕㄨˇ'], '數數': ['ㄕㄨˇ', 'ㄕㄨˇ'], '一起數': ['ㄧ', 'ㄑㄧˇ', 'ㄕㄨˇ'], '只數': ['ㄓˇ', 'ㄕㄨˇ'],
    '數到': ['ㄕㄨˇ', 'ㄉㄠˋ'], '數完': ['ㄕㄨˇ', 'ㄨㄢˊ'], '數出': ['ㄕㄨˇ', 'ㄔㄨ'], '下數': ['ㄒㄧㄚˋ', 'ㄕㄨˇ'], '數看': ['ㄕㄨˇ', 'ㄎㄢˋ'], '倒數': ['ㄉㄠˋ', 'ㄕㄨˇ'],
    '慢慢地': ['ㄇㄢˋ', 'ㄇㄢˋ', '˙ㄉㄜ'], '快快地': ['ㄎㄨㄞˋ', 'ㄎㄨㄞˋ', '˙ㄉㄜ'], '輕輕地': ['ㄑㄧㄥ', 'ㄑㄧㄥ', '˙ㄉㄜ'], '用力地': ['ㄩㄥˋ', 'ㄌㄧˋ', '˙ㄉㄜ'], '好好地': ['ㄏㄠˇ', 'ㄏㄠˇ', '˙ㄉㄜ'],
    '轉圈': ['ㄓㄨㄢˋ', 'ㄑㄩㄢ'], '轉一圈': ['ㄓㄨㄢˋ', 'ㄧ', 'ㄑㄩㄢ'], '骰子': ['ㄕㄞˇ', '˙ㄗ'], '擲骰子': ['ㄓˊ', 'ㄕㄞˇ', '˙ㄗ'],
    '哪裡': ['ㄋㄚˇ', 'ㄌㄧˇ'], '差一點': ['ㄔㄚ', 'ㄧ', 'ㄉㄧㄢˇ'], '可調': ['ㄎㄜˇ', 'ㄊㄧㄠˊ'], '重錄': ['ㄔㄨㄥˊ', 'ㄌㄨˋ'], '慢慢數': ['ㄇㄢˋ', 'ㄇㄢˋ', 'ㄕㄨˇ'], '聲數': ['ㄕㄥ', 'ㄕㄨˇ'], '種的花': ['ㄓㄨㄥˋ', '˙ㄉㄜ', 'ㄏㄨㄚ'], '一樣': ['ㄧ', 'ㄧㄤˋ'], '一個': ['ㄧ', 'ㄍㄜˋ'], '一下': ['ㄧ', 'ㄒㄧㄚˋ'], '一起': ['ㄧ', 'ㄑㄧˇ'], '一點': ['ㄧ', 'ㄉㄧㄢˇ'],
}


def fmt(r):
    return '˙' + r[:-1] if r.endswith('˙') else r


def moe_words(moe):
    """教育部辭典：詞 → 每個字的注音（只收讀音格式正確、字數相符的第一個讀音）"""
    out = {}
    for w, rs in moe.items():
        for r in rs:
            toks = r.split()
            if len(toks) == len(w) and all(ZY.match(t) for t in toks):
                out[w] = toks; break
    return out


def char_readings(moe, ch):
    return [r for r in moe.get(ch, []) if ZY.match(r)]


def read_run(run, moe, words):
    s = T2S.convert(run)
    if len(s) != len(run):
        s = run
    out = [fmt(x[0]) for x in pinyin(s, style=Style.BOPOMOFO)]
    src = ['py'] * len(run)
    for i, ch in enumerate(run):
        ok = char_readings(moe, ch)
        if ok and out[i] not in ok:
            out[i] = ok[0]; src[i] = 'moe-char'
        if ch in DEFAULT_PREF:
            out[i] = DEFAULT_PREF[ch]; src[i] = 'pref'
    bounds, k = {0}, 0
    for tok in jieba.cut(s, HMM=False):
        k += len(tok); bounds.add(k)
    i = 0
    while i < len(run):
        for L in range(min(MAXW, len(run) - i), 1, -1):
            w = run[i:i + L]
            if w in words and i in bounds and i + L in bounds:
                out[i:i + L] = words[w]; src[i:i + L] = ['moe-word'] * L
                i += L - 1; break
        i += 1
    for i, ch in enumerate(run):
        if ch in FORCE:
            out[i] = FORCE[ch]; src[i] = 'force'
    for ph, rs in PHRASE.items():
        for m in re.finditer(re.escape(ph), run):
            for k, r in enumerate(rs):
                out[m.start() + k] = r; src[m.start() + k] = 'phrase'
    return out, src


def word_bounds(run):
    s = T2S.convert(run)
    s = s if len(s) == len(run) else run
    b, k = {0}, 0
    for tok in jieba.cut(s, HMM=False):
        k += len(tok); b.add(k)
    return b


def build(text, moe, report=False):
    words = moe_words(moe)
    runs = CJK.findall(text)
    per = defaultdict(Counter)
    used_words = {}
    occ = []
    for run in runs:
        rs, src = read_run(run, moe, words)
        for i, (ch, r) in enumerate(zip(run, rs)):
            if r and ZY.match(r):
                per[ch][r] += 1
                occ.append((run, i, r, src[i]))
    d = {ch: c.most_common(1)[0][0] for ch, c in per.items()}
    # 畫面上用的詞表：教育部的詞、課本規則詞，以及「和這個字常用讀音不同」的前後文
    for run in runs:
        rs, src = read_run(run, moe, words)
        bd = word_bounds(run)
        i = 0
        while i < len(run):
            hit = 0
            for L in range(min(MAXW, len(run) - i), 1, -1):
                w = run[i:i + L]
                if w in PHRASE or (w in words and i in bd and i + L in bd):
                    if any(rs[i + k] != d.get(run[i + k]) for k in range(L)):
                        used_words[w] = rs[i:i + L]
                    hit = L; break
            if not hit and d.get(run[i]) and rs[i] != d[run[i]]:
                for w, k0 in ((run[i - 1:i + 1], 1) if i else (None, 0), (run[i:i + 2], 0) if i + 1 < len(run) else (None, 0)):
                    if w and len(w) == 2 and w not in used_words:
                        used_words[w] = rs[i - k0:i - k0 + 2]
            i += max(hit - 1, 0) + 1
    for ph, rs in PHRASE.items():
        used_words[ph] = rs
    z = {'d': d, 'w': used_words}
    if report:
        poly = {ch for ch in d if len(char_readings(moe, ch)) > 1}
        unchecked = Counter((run[max(0, i - 2):i + 3], run[i], r) for run, i, r, s in occ if run[i] in poly and s in ('py', 'moe-char'))
        return z, unchecked
    return z


def lookup(z, t):
    out, i = [], 0
    while i < len(t):
        for L in range(min(MAXW, len(t) - i), 1, -1):
            if t[i:i + L] in z['w']:
                out += z['w'][t[i:i + L]]; i += L; break
        else:
            out.append(z['d'].get(t[i], '?')); i += 1
    return out


if __name__ == '__main__':
    if sys.argv[1] == '--trim':
        text = open(sys.argv[2], encoding='utf-8').read(); full = json.load(open(sys.argv[3], encoding='utf-8'))
        need = set()
        for run in CJK.findall(text):
            for i in range(len(run)):
                for L in range(1, MAXW + 1):
                    if i + L <= len(run): need.add(run[i:i + L])
        trim = {k: v for k, v in full.items() if k in need}
        json.dump(trim, open('zhuyin_moe.json', 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
        print('trimmed', len(trim), 'entries')
    else:
        moe = json.load(open('zhuyin_moe.json', encoding='utf-8'))
        z, unchecked = build(open(sys.argv[1], encoding='utf-8').read(), moe, report=True)
        print(len(z['d']), 'chars,', len(z['w']), 'words;', len(unchecked), 'polyphone contexts not covered by MOE words')
        for t in sys.argv[2:]:
            print(t, ' '.join(lookup(z, t)))
