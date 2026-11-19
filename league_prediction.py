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
        inp = input('NHẬP 5 TƯỚNG: ').strip()
        if not inp: continue
        champs = [c.strip() for c in inp.split(',') if c.strip()]
        if len(champs) != 5:
            print('NHẬP ĐỦ 5 TƯỚNG CÁCH NHAU BẰNG DẤU PHẨY!\n')
            continue
        miss = [c for c in champs if proc(c) not in lookup]
        if miss:
            print('KHÔNG TÌM THẤY TƯỚNG:', ','.join(miss), '!\n')
            continue
        processed_champs = [champ_info[lookup[proc(c)]]['name'] for c in champs]
        return processed_champs

def recommender(my_team, opp_team, my_side='blue', role=None, top_k = 5):
    my_riotid = [lookup[proc(c)] for c in my_team if proc(c) in lookup]
    opp_riotid = [lookup[proc(c)] for c in opp_team if proc(c) in lookup]

    chosen = set(my_riotid + opp_riotid)

    # Dự đoán tỉ lệ thắng hiện tại
    my_val = 1
    opp_val = -1
    if my_side != 'blue': my_val, opp_val = opp_val, my_val
    cur = np.zeros((1, champ_count), dtype = np.int32)
    for riotid in my_riotid: cur[0][riotid_to_idx[riotid]] = my_val
    for riotid in opp_riotid: cur[0][riotid_to_idx[riotid]] = opp_val
    cur_winrate = model.predict_proba(cur)[0, 1]

    # Dự đoán tỉ lệ thắng từng tướng
    candidates = []
    for riotid in sorted_id:
        if riotid in chosen: continue

        info = champ_info[riotid]
        tag = info.get('tags', [])

        if role and role.lower() != 'all':
            if not any(role.lower() in t.lower() for t in tag):
                continue

        cur_copy = cur.copy()
        cur_copy[0][riotid_to_idx[riotid]] = my_val
        new_winrate = model.predict_proba(cur_copy)[0, 1]

        delta = new_winrate - cur_winrate

        candidates.append({
            'name' : info['name'],
            'tag' : ','.join(tag),
            'new_winrate' : new_winrate,
            'delta' : delta
        })

    # Sắp xếp
    candidates.sort(key = lambda x: x['delta'], reverse = True)
    best_pick = candidates[:top_k]
    worst_pick = sorted(candidates, key=lambda x: x['delta'])[:3]
    side_label = "🔵 ĐỘI XANH" if my_side == 'blue' else "🔴 ĐỘI ĐỎ"

    print("\n" + "=" * 65)
    print('Phe của bạn:', side_label)
    print('Đội bạn đã chọn:', ','.join(my_team) if my_team else '(Chưa chọn tướng nào)')
    print('Đội địch đã chọn:', ','.join(opp_team) if opp_team else '(Chưa chọn tướng nào)')
    if role and role.lower() != 'all':
        print('Hệ tướng yêu cầu:', role)
    print('Tỉ lệ thắng hiện tại:', str(round(cur_winrate * 100, 2)) + '%')
    print("=" * 65)
    print('Top 5 tướng tối ưu nên pick:')
    for id, c in enumerate(best_pick, 1):
        print(str(id) + '/', c['name'], '-->', 'Tỉ lệ thắng mới:', str(round(c['new_winrate'] * 100, 2)) + '%')
    print('Top 3 tướng cần tránh pick:')
    for id, c in enumerate(worst_pick, 1):
        print(str(id) + '/', c['name'], '-->', 'Tỉ lệ thắng mới:', str(round(c['new_winrate'] * 100, 2)) + '%')
    print("=" * 65 + "\n")

def input_recommender():
    print("Chọn phe của bạn:")
    print("  [1] 🔵 Đội Xanh (Blue Team)")
    print("  [2] 🔴 Đội Đỏ (Red Team)")
    inp = input('Nhập lựa chọn của bạn:').strip()
    my_side = 'blue' if inp == '1' else 'red'

    # Nhập tướng đội mình
    while True:
        print('Nhập tướng đội bạn đã chọn! (tối đa 4 tướng, cách nhau bởi dấu phẩy): ')
        my_inp = input('Tướng đội bạn: ').strip()
        my_pick = [c.strip() for c in my_inp.split(',') if c.strip()]
        if len(my_pick) > 4:
            print('Chỉ nhập tối đa 4 tướng!')
            continue
        miss = [proc(c) for c in my_pick if proc(c) not in lookup]
        if miss:
            print('Không tìm thấy tướng:', ', '.join(miss), '!')
            continue
        my_pick = [champ_info[lookup[proc(c)]]['name'] for c in my_pick]
        break

    # Nhập tướng đội địch
    while True:
            print('Nhập tướng đội địch đã chọn! (tối đa 4 tướng, cách nhau bởi dấu phẩy): ')
            opp_inp = input('Tướng đội địch: ').strip()
            opp_pick = [c.strip() for c in opp_inp.split(',') if c.strip()]
            if len(opp_pick) > 5:
                print('Chỉ nhập tối đa 5 tướng!')
                continue
            miss = [proc(c) for c in opp_pick if proc(c) not in lookup]
            if miss:
                print('Không tìm thấy tướng:', ', '.join(miss), '!')
                continue
            opp_pick = [champ_info[lookup[proc(c)]]['name'] for c in opp_pick]
            break

    lap = set(my_pick).intersection(set(opp_pick))
    if lap:
        print('Có tướng bị trùng giữa 2 phe:', ', '.join(lap), '!', 'Vui lòng nhập lại.')
        return

    print('Lọc theo vai trò tướng cần tìm:')
    print('  [0] Tất cả hệ (All)')
    print('  [1] Fighter (Đấu Sĩ)')
    print('  [2] Mage (Pháp Sư)')
    print('  [3] Marksman (Xạ Thủ)')
    print('  [4] Assassin (Sát Thủ)')
    print('  [5] Tank (Đỡ Đòn)')
    print('  [6] Support (Hỗ Trợ)')
    choice = input('Nhập lựa chọn của bạn:').strip()
    mapping = {'1': 'Fighter',
               '2': 'Mage',
               '3': 'Marksman',
               '4': 'Assassin',
               '5': 'Tank',
               '6': 'Support'}
    role = mapping.get(choice, None)

    recommender(my_pick, opp_pick, my_side, role, top_k = 5)
    
while True:
    print('=' * 65)
    print('CHỌN CHỨC NĂNG:')
    print('  [1] DỰ ĐOÁN TỈ LỆ THẮNG 5 VS 5')
    print('  [2] GỢI Ý LƯỢT PICK TỐI ƯU')
    print('  [3] THOÁT')
    choice = input('NHẬP LỰA CHỌN CỦA BẠN (1, 2 hoặc 3):').strip()

    if choice == '3':
        print('TẠM BIỆT!')
        break

    elif choice == '1':
        blue_team = input_team('ĐỘI XANH')
        red_team = input_team('ĐỘI ĐỎ')
        pred(blue_team, red_team)

    elif choice == '2':
        input_recommender()

    else: print('LỰA CHỌN KHÔNG HỢP LỆ!')
