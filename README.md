# LMCOR Training and Inference System

Hệ thống training và inference cho mô hình LMCOR với hỗ trợ nhiều task, backbone và mode.

## Giới thiệu

Đây là implementation của bài báo **"Small Language Models Improve Giants by Rewriting Their Outputs"** trên dataset tiếng Việt. Hệ thống sử dụng các mô hình ngôn ngữ nhỏ (small language models) để cải thiện chất lượng output của các mô hình lớn hơn thông qua kỹ thuật rewriting.

Ngoài code chính, nhóm cũng đã tổng hợp lại code thành folder `notebooks/` chứa các notebook Jupyter, cho phép người dùng chạy thực nghiệm trên các nền tảng cloud (như Google Colab, Kaggle, etc.) mà không cần cài đặt môi trường local.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

### Training

```bash
python main.py train \
    --task MT \
    --backbone google/mt5-base \
    --mode single \
    --train_file path/to/train.csv \
    --epochs 3 \
    --learning_rate 3e-4
```

### Inference

Có 2 cách để chạy inference:

**Cách 1: Sử dụng task/backbone/mode (tự động tìm model)**
```bash
python main.py infer \
    --task MT \
    --backbone google/mt5-base \
    --mode single \
    --test_file path/to/test.csv
```

**Cách 2: Chỉ định đường dẫn model trực tiếp**
```bash
python main.py infer \
    --model_path ./models/mt/google_mt5-base/single/final_model \
    --test_file path/to/test.csv
```

## Tham số

### Task
- `MT`: Machine Translation (đánh giá bằng BLEU)
- `GEC`: Grammar Error Correction (đánh giá bằng ERRANT - Precision, Recall, F0.5)

### Backbone
- `google/mt5-base`
- `VietAI/vit5-base`
- Hoặc bất kỳ model Seq2Seq nào từ HuggingFace

### Mode
- `single`: Sử dụng 1 candidate (candidate_1 cho MT, candidate_3 cho GEC)
- `multi`: Sử dụng 3 candidates (candidate_1, candidate_2, candidate_3)

## Lưu và Load Model

### Khi Training
- Model được lưu tại: `{model_dir}/{task}/{backbone}/{mode}/final_model/`
- Checkpoints được lưu tại: `{model_dir}/{task}/{backbone}/{mode}/checkpoints/` (nếu bật save_checkpoints)
- Metadata training được lưu trong file `training_metadata.json` cùng với model

### Khi Inference
- Có thể load model bằng cách chỉ định `--model_path` trực tiếp
- Hoặc để hệ thống tự động tìm model dựa trên `task/backbone/mode`
- Hệ thống sẽ tự động đọc metadata để xác định task/mode nếu có

## Cấu trúc Output

### Predictions
Kết quả prediction được lưu tại:
```
{task}/prediction/{backbone}/{mode}/{test_filename}.csv
```

### Reports
Báo cáo đánh giá được lưu tại:
```
{task}/report/{backbone}/{mode}/{test_filename}.txt
```

### Models
Model được lưu tại:
```
{model_dir}/{task}/{backbone}/{mode}/final_model/
```

## Cấu trúc dữ liệu

### Task: Machine Translation (MT)

#### File CSV cho Training/Test - Task MT

File CSV cho task MT cần có các cột sau:

**Các cột bắt buộc:**
- `en` hoặc `input`: Văn bản nguồn (tiếng Anh)
- `vi` hoặc `output`: Văn bản đích (reference translation - tiếng Việt)
- `candidate_1`: Candidate translation thứ nhất (bắt buộc cho mode `single`)
- `candidate_2`: Candidate translation thứ hai (bắt buộc cho mode `multi`)
- `candidate_3`: Candidate translation thứ ba (bắt buộc cho mode `multi`)

**Ví dụ cấu trúc file `mt_train.csv` hoặc `mt_test.csv`:**

```csv
en,vi,candidate_1,candidate_2,candidate_3
"Hello, how are you?","Xin chào, bạn khỏe không?","Chào bạn, bạn thế nào?","Xin chào, bạn có khỏe không?","Chào, bạn khỏe chứ?"
"The weather is nice today.","Thời tiết hôm nay đẹp.","Hôm nay thời tiết đẹp.","Thời tiết hôm nay rất đẹp.","Hôm nay trời đẹp."
```

**Lưu ý:**
- Mode `single`: Chỉ sử dụng `candidate_1`
- Mode `multi`: Sử dụng cả 3 candidates (`candidate_1`, `candidate_2`, `candidate_3`)

---

### Task: Grammar Error Correction (GEC)

#### File CSV cho Training/Test - Task GEC

File CSV cho task GEC cần có các cột sau:

**Các cột bắt buộc:**
- `en` hoặc `input`: Văn bản nguồn có lỗi ngữ pháp (tiếng Việt)
- `output` hoặc `vi`: Văn bản đích đã được sửa lỗi (reference correction - tiếng Việt)
- `candidate_1`: Candidate correction thứ nhất (bắt buộc cho mode `multi`)
- `candidate_2`: Candidate correction thứ hai (bắt buộc cho mode `multi`)
- `candidate_3`: Candidate correction thứ ba (bắt buộc cho cả mode `single` và `multi`)

**Ví dụ cấu trúc file `gec_train.csv` hoặc `gec_test.csv`:**

```csv
imput,output,candidate_1,candidate_2,candidate_3
"Tôi ik học vào ngay mai.","Tôi sẽ đi học vào ngày mai.","Tôi sẽ đi học ngày mai.","Tôi đi học vào ngày mai nhé.","Tôi sẽ đi học vào ngày mai."
"Hôm kua tôi đã ăn kơm.","Hôm qua tôi đã ăn cơm.","Hôm qua tôi ăn cơm.","Tôi đã ăn cơm hôm qua.","Hôm qua tôi đã ăn cơm."
```

**Lưu ý:**
- Mode `single`: Sử dụng `candidate_3` (hoặc `candidate_1` nếu `candidate_3` không có)
- Mode `multi`: Sử dụng cả 3 candidates (`candidate_1`, `candidate_2`, `candidate_3`)

## Chạy trên Cloud Platforms

Để chạy thực nghiệm trên các nền tảng cloud mà không cần cài đặt môi trường local, bạn có thể sử dụng các notebook trong folder `notebooks/`:

- `MT_Experiment.ipynb`: Notebook tổng hợp cho thực nghiệm Machine Translation
- `GEC_mt5_single.ipynb`: Thực nghiệm với mT5-base, single mode
- `GEC_mt5_multi.ipynb`: Thực nghiệm với mT5-base, multi mode
- `GEC_mt5_single.ipynb`: Thực nghiệm với ViT5-base, single mode
- `GEC_mt5_multi.ipynb`: Thực nghiệm với ViT5-base, multi mode

Các notebook này có thể chạy trực tiếp trên Google Colab, Kaggle, hoặc các nền tảng Jupyter notebook khác.

## Ví dụ

### Training MT model với mT5-base, single mode
```bash
python main.py train --task MT --backbone google/mt5-base --mode single --train_file data/mt_train.csv
```

### Inference GEC model với ViT5-base, multi mode
```bash
python main.py infer --task GEC --backbone VietAI/vit5-base --mode multi --test_file data/gec_test.csv
```

