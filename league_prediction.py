#import thư viện

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier # Sử dụng Gradient Boosting
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, RocCurveDisplay, ConfusionMatrixDisplay

print("=" * 65)
print('ĐANG CHUẨN BỊ NHỮNG FILE CẦN THIẾT...')
print("=" * 65)
# Đọc file lịch sử đấu
df = pd.read_csv('matches.csv')

# Đọc file metadata tướng
with open('champions.json', 'r') as champ:
    champ_info = {int(k): v for k, v in json.load(champ).items()}
print('ĐÃ LOAD NHỮNG FILE CẦN THIẾT!')
print("=" * 65)

# Do id tướng không theo thứ tự -> tạo index từ 0 đến n-1
sorted_id = sorted(list(champ_info.keys()))
riotid_to_idx = {riotid: idx for idx, riotid in enumerate(sorted_id)}
idx_to_riotid = {idx:riotid for riotid, idx in riotid_to_idx.items()}
champ_count = len(sorted_id)


# Gán nhãn 1 cho đội xanh, -1 cho đội đỏ
blue_cols = ['blue_top_id', 'blue_jg_id', 'blue_mid_id', 'blue_bot_id', 'blue_sup_id']
red_cols = ['red_top_id', 'red_jg_id', 'red_mid_id', 'red_bot_id', 'red_sup_id']
X = np.zeros((len(df), champ_count), dtype=np.int32)
y = df['blue_win'].values.astype(int)
for rowid, (_, row) in enumerate(df.iterrows()):
    for c in blue_cols:
        X[rowid, riotid_to_idx[row[c]]] = 1
    for c in red_cols:
        X[rowid, riotid_to_idx[row[c]]] = -1

# Chia train, test
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size = 0.2, random_state = 42, stratify=y
)

# Nhân đôi dữ liệu: A xanh B đỏ: xanh thắng -> A đỏ B xanh: đỏ thắng
# => thêm một nửa tập dữ liệu X <=> -X, y <=> 1-y
X_train = np.vstack([X_train_raw, -X_train_raw])
y_train = np.concatenate([y_train_raw, 1-y_train_raw])

X_test = np.vstack([X_test_raw, -X_test_raw])
y_test = np.concatenate([y_test_raw, 1-y_test_raw])

# Train model
model = HistGradientBoostingClassifier(
    max_iter = 250,
    learning_rate = 0.02,
    max_depth = 6,
    min_samples_leaf = 25,
    l2_regularization = 1.5,
    early_stopping = True,
    n_iter_no_change = 10,
    random_state = 42
)

print('ĐANG HUẤN LUYỆN MÔ HÌNH...')
print("=" * 65)
model.fit(X_train, y_train)
print('ĐÃ HOÀN TẤT HUẤN LUYỆN MÔ HÌNH! HÃY BẮT ĐẦU NHẬP ĐỘI HÌNH:')
print("=" * 65)

# Đánh giá
"""
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
loss = log_loss(y_test, y_pred_proba)

print(acc * 100)
print(auc)
print(loss)
"""
# Dự đoán team 5 vs 5
proc = lambda s: s.lower().replace(" ", "").replace("'", "").replace(".", "")

lookup = {}

for riotid, info in champ_info.items():
    lookup[proc(info['name'])] = riotid
    lookup[proc(info['key'])] = riotid

def pred(blue_team, red_team):

    if len(blue_team) != 5 or len(red_team) != 5:
        print('Mỗi đội đúng 5 tướng!')
        return
    
    for riotid, info in champ_info.items():
        lookup[proc(info['name'])] = riotid
        lookup[proc(info['key'])] = riotid

    stat = np.zeros((1, champ_count), dtype=np.int32)

    try:
        for champ_name in blue_team:
            riotid = lookup[proc(champ_name)]
            stat[0][riotid_to_idx[riotid]] = 1
        for champ_name in red_team:
            riotid = lookup[proc(champ_name)]
            stat[0][riotid_to_idx[riotid]] = -1
    except KeyError as e:
        print('Không thấy tướng tên:', e)
        return

    prob_blue = model.predict_proba(stat)[0, 1]
    print("\n" + "=" * 65)
    prob_red = 1 - prob_blue
    print('⚔️ KẾT QUẢ DỰ ĐOÁN TỈ LỆ THẮNG ⚔️')
    print("=" * 65)
    print('🔵 ĐỘI XANH:', ', '.join(blue_team))
    print('🔴 ĐỘI ĐỎ:', ', '.join(red_team))

    typ = '🔵 ĐỘI XANH CÓ LỢI THẾ 🔵' if prob_blue >= 0.5 else '🔴 ĐỘI ĐỎ CÓ LỢI THẾ 🔴'
    
    print('🏆 ĐÁNH GIÁ:', typ)
    print('   •TỈ LỆ THẮNG ĐỘI XANH:', str(round(prob_blue * 100, 2)) + '%')
    print('   •TỈ LỆ THẮNG ĐỘI ĐỎ:', str(round(prob_red * 100, 2)) + '%')
    print("=" * 65 + "\n")

def input_team(team):
    while True:
        print('NHẬP 5 TƯỚNG', team, '(CÁCH NHAU DẤU PHẨY):')
        inp = input('Nhập 5 tướng: ').strip()
        if not inp: continue
        champs = [c.strip() for c in inp.split(',') if c.strip()]
        if len(champs) != 5:
            print('Nhập đủ 5 tướng cách nhau bằng dấu phẩy!\n')
            continue
        miss = [c for c in champs if proc(c) not in lookup]
        if miss:
            print('Không tìm thấy tướng:', ','.join(miss), '!\n')
            continue
        processed_champs = [champ_info[lookup[proc(c)]]['name'] for c in champs]
        return processed_champs
    
print("=" * 65)
print('NHẬP ĐỘI HÌNH:')
while True:
    blue_team = input_team('ĐỘI XANH')
    red_team = input_team('ĐỘI ĐỎ')

    pred(blue_team, red_team)

    typ = input("DỰ ĐOÁN TRẬN KHÁC? (GÕ 'q' HOẶC 'k' ĐỂ THOÁT, ENTER ĐỂ TIẾP TỤC)").strip().lower()
    if typ in ['q', 'quit', 'exit', 'k']:
        print('TẠM BIỆT!')
        break
    print("-" * 65 + '\n')

