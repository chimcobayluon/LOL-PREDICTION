# DỰ ĐOÁN TỈ LỆ THẮNG GIỮA 2 TEAM COMP LOL

Một project nhỏ dự đoán tỉ lệ thắng của 2 bên xanh và đỏ dựa trên các tướng của 2 bên

# Cách nó hoạt động

Dữ liệu của mô hình được tổng hợp từ 2 file:
- champion.json: Lưu thông tin của tướng
- matches.csv*: Lưu thông tin của trận đấu

Mô hình được học từ lịch sử đấu trong matches.csv về số trận thắng của các tướng, team comp, đội xanh/đỏ

Trong từng trận các tướng của đội xanh được gán nhãn 1, đội đỏ gán nhãn -1 và các tướng khác gán nhãn 0, dự liệu được nhân đôi khiến bộ dữ liệu được tăng lên khoảng 40.000 trận vì lí do sau: A đỏ B xanh, xanh thắng <=> A xanh B đỏ, đỏ thắng

Mô hình sử dụng: Gradient Boosting

Lí do sử dụng: Decision Trees có khả năng học một cách phi tuyến, ví dụ: cặp đôi phối hợp tốt Yasuo + Malphite, kèo khắc chế Master Yi >< Rammus

# Set up

```
pip install -r requirements.txt
python/python3 league_prediction.py
```

# Data

Được lấy từ API của RIOT, dựa trên khoảng 20.000 trận đấu của rank Thách đấu Hàn để dữ liệu có thể phản ánh chính xác nhất chất tướng

# License

Làm mọi thứ bạn thích với nó

