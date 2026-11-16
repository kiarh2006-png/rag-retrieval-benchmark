# Advanced Retrieval Benchmark

Benchmark مستقل برای مقایسه Dense، BM25 و Hybrid-RRF روی Corpus ثابت فارسی.

## وضعیت داده
- Corpus: 5,782 chunk از 20 سند
- Embeddings: ماتریس `data/embeddings/dense_embeddings.npy` با شکل `(5782, 1024)` و `float32`
- مدل embedding گزارش‌شده در manifest: `bge-m3`
- Embeddings دوباره محاسبه نشده‌اند؛ ترتیب ردیف‌ها با `chunks.jsonl` هم‌تراز است.
- Ground Truth فعلی: 100 سؤال، 90 پاسخ‌پذیر و 10 بدون پاسخ.

## نصب و آماده‌سازی (Windows PowerShell)
```powershell
cd C:\Users\Rohi\rag-retrieval-benchmark
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

برای Dense و Hybrid، Ollama باید روشن باشد و مدل embedding با مدل سازنده ماتریس Corpus یکسان باشد:
```powershell
ollama serve
```
در پنجره دیگری:
```powershell
ollama list
ollama pull bge-m3
```
اگر `bge-m3` از قبل نصب است، pull لازم نیست. API مورد استفاده `http://localhost:11434/api/embed` است. Unit testها به Ollama وابسته نیستند.

## اجرای کنترل کیفیت
```powershell
python scripts/validate_ground_truth.py
pytest
```
`pytest.ini` فقط پوشه `tests/` را جمع‌آوری می‌کند؛ فایل‌های `scripts/probe_*.py` ابزارهای دستی/Integration هستند و در Unit Test اجرا نمی‌شوند.

## اجرای Benchmark
فرمان اصلی:
```powershell
python scripts/run_benchmark.py --config configs/baseline.yaml
```
نتیجه در `results/baseline.json` ذخیره می‌شود. فایل خروجی شامل تنظیمات، نتیجه هر سؤال، خلاصه کلی، خلاصه دسته‌بندی‌شده و کاندیداهای شکست است.

## آزمایش‌های پارامتری
سه اجرای کنترل‌شده (هر بار یک پارامتر):
```powershell
python scripts/run_experiments.py
```
- baseline: `top_k=5`, `candidate_k=100`, `rrf_k=60`, وزن‌های 1:1
- `topk_10`: فقط `top_k` از 5 به 10 تغییر می‌کند.
- `rrf_k_20`: فقط پارامتر RRF از 60 به 20 تغییر می‌کند.

آزمایش وزن‌دهی RRF (فقط وزن Dense/BM25 از 1:1 به 2:1):
```powershell
python scripts/run_benchmark.py --config configs/weighted_rrf.yaml
```

در تمام آزمایش‌ها Corpus، Embeddings و Ground Truth باید ثابت بمانند. فایل config همراه هر نتیجه در فیلد `metadata.settings` ذخیره می‌شود.

## روش‌های Retrieval
- Dense: ضرب ماتریس embedding نرمال‌شده در embedding پرس‌وجو (cosine similarity).
- Lexical: BM25Okapi با نرمال‌سازی حروف عربی/فارسی و ارقام فارسی/عربی.
- Hybrid: Reciprocal Rank Fusion؛ برای هر chunk امتیاز `sum(weight / (rrf_k + rank))` محاسبه می‌شود. chunk تکراری فقط یک‌بار در خروجی می‌آید.

## تعریف Metricها
برای هر پرسش پاسخ‌پذیر، مجموعه مرتبط‌ها `R` و فهرست بازیابی‌شده مرتب‌شده `L` است:
- Hit@K: آیا حداقل یک chunk مرتبط در K رتبه اول وجود دارد؟ مقدار دودویی 0 یا 1.
- Recall@5: تعداد chunkهای مرتبط بازیابی‌شده در پنج رتبه اول تقسیم بر تعداد کل chunkهای مرتبط.
- MRR: معکوس رتبه اولین chunk مرتبط؛ اگر هیچ‌کدام پیدا نشود صفر.
- nDCG@5: بهره تجمعی رتبه‌بندی‌شده در پنج رتبه اول نسبت به رتبه‌بندی ایده‌آل، با gain دودویی.

مثال واقعی از Ground Truth: برای q001 دو chunk مرتبط `taazirat-0130` و `taazirat-0131` تعریف شده‌اند. اگر سیستم فقط یکی را در Top-5 بیاورد، Hit@5 برابر 1 است چون دست‌کم یک شاهد پیدا شده، اما Recall@5 برابر 0.5 است چون فقط یکی از دو شاهد بازیابی شده است.

محدودیت‌ها: Hit@K تعداد شواهد را نشان نمی‌دهد؛ Recall به کامل بودن Ground Truth وابسته است؛ MRR فقط رتبه اولین شاهد را می‌سنجد؛ nDCG با برچسب دودویی، درجات ارتباط را مدل نمی‌کند.

سؤال‌های بدون پاسخ در شش Metric رتبه‌بندی وارد میانگین نمی‌شوند، چون مجموعه مرتبط ندارند. تعداد آن‌ها جداگانه گزارش می‌شود. این پروژه هنوز معیار کالیبره‌شده برای abstention/رد پاسخ را پیاده‌سازی نمی‌کند.

## کیفیت Ground Truth
روش بررسی و توزیع دسته‌ها در `data/eval/ground_truth_review.md` ثبت شده است. اسکریپت Validator ساختار، شناسه‌ها و تعلق chunk به سند را بررسی می‌کند؛ این تست ساختاری جایگزین بازبینی معنایی انسانی نیست. سؤال‌های جدید q037-q090 با متن chunkهای ارجاع‌شده تطبیق داده شده‌اند. سؤال‌های بدون پاسخ q091-q100 به‌عنوان نمونه‌های خارج از Corpus تعریف شده‌اند و توصیه می‌شود یک بازبین دوم آن‌ها را هم کنترل کند.

## تحلیل نتایج و محدودیت فعلی
- نتایج آرشیوی 20 سؤال اول در `data/eval/weighted_comparison.json` ثبت شده‌اند؛ تحلیل مهندسی در `data/eval/engineering_analysis.md` و 15 شکست تاریخی در `data/eval/failure_analysis_historical_20q.md` آمده است. این‌ها نتیجه نهایی دیتاست 100 سؤالی نیستند.
- یک اجرای آفلاین BM25 روی 100 سؤال در `results/bm25_fallback_smoke_100.json` و خلاصه آن در `data/eval/current_100q_partial_results.md` نگهداری شده است. چون در محیط ساخت این بسته `rank_bm25` نصب نبود، از پیاده‌سازی fallback سازگار با فرمول Okapi استفاده شد؛ این فایل Smoke Test است و نباید به‌عنوان نتیجه نهایی سه‌روش یا جایگزین اجرای محلی تلقی شود.
- اجرای کامل Dense/Hybrid در این محیط ممکن نبود، چون سرویس Ollama در دسترس نبود. اجرای Benchmark روی دستگاه هدف با Ollama روشن، خروجی نهایی را تولید می‌کند.
- برای گزارش 15 شکست نهایی، از `failure_candidates` خروجی Benchmark استفاده کن و متن retrieved را با Ground Truth مقایسه کن؛ root cause را بدون بازبینی دستی به‌صورت خودکار قطعی اعلام نکن.

## ساختار
```text
configs/                  تنظیمات baseline و آزمایش‌ها
data/corpus/              چانک‌های ثابت
data/embeddings/          Embeddingهای ثابت Corpus
data/eval/                Ground Truth، گزارش‌ها و Validator
src/retrieval/            Dense، BM25 و Hybrid-RRF
src/evaluation/           Metricها و گزارش دسته‌بندی
scripts/                  اجرای Benchmark و ابزارهای تشخیصی
tests/                    Unit Test مستقل از Ollama
results/                  خروجی آزمایش‌ها
```
