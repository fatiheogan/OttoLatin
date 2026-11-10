# OttoLatin – Veri (Team 11)

Hazırlayan: Fatih (veri sorumlusu) · 8 Kasım 2026

## Dosyalar

| Dosya | Satır | Ne için |
|---|---|---|
| `data/processed/train.jsonl` | 2.417.982 | Tam eğitim seti |
| `data/processed/train_small.jsonl` | 300.000 | Hızlı baseline / mid sunum eğitimi (train'den rastgele, seed 42) |
| `data/processed/dev.jsonl` | 25.357 | Model seçimi, erken durdurma |
| `data/processed/test.jsonl` | 26.534 | İç test (eğitimle aynı dağılım) |
| `data/external/riqa.jsonl` | 1.575 | Dış test – gerçek el yazısı (Rika) belgeler |
| `data/external/naskh.jsonl` | 376 | Dış test – gerçek matbu (Nesih) kitap |
| `data/external/ot2tr.jsonl` | 2.358 | Dış test – 100 sayfa arşiv belgesi, ferman, roman |

**Format (jsonl, UTF-8):**
- processed: `{"id", "ottoman", "turkish"}`
- external: `{"id", "source", "ottoman", "turkish", "turkish_norm", "ottoman_bytes"}`

## Kaynaklar

- **Eğitim:** [Kotek37/ottoman-turkish-4m](https://huggingface.co/datasets/Kotek37/ottoman-turkish-4m), 2.717.146 satır, tek split. Lisans ve üretim yöntemi belirtilmemiş (sahibine soruldu / sorulacak).
- **Dış test:** [osmanlica.com/test](https://osmanlica.com/test) (Mina ARGE). Lisans belirtilmemiş, "araştırma ve akademik amaçlı" kullanım; sürüm ve tarih ile atıf istiyor.
- **Kullanılmadı:** Zenodo Osmanlica-Bench-v1. Sayfada 6.500 örnek yazıyor ama arşivde (3,6 kB) yalnızca birkaç satırlık örnek CSV var.

## Eğitim verisi hakkında bilinmesi gerekenler

1. **Sentetik.** Türkçe taraf modern Türkçe (Wikipedia ve altyazı benzeri: "Orhan Pamuk Masumiyet Müzesi", "Prostat kanseri"). Osmanlıca taraf büyük ihtimalle otomatik çeviri: aynı Türkçe cümlenin farklı Osmanlıca yazımı 2,7M satırda yalnızca 134 kez geçiyor. Araç hataları da görünüyor ("kes" → "كثع", "km" → "قم").
2. **Satırlar ortak bir cümle havuzundan birleştirilmiş.** Rastgele split yapıldığında test cümlelerinin **%77,5'i** train'de de geçiyordu. Test ve dev cümlelerini içeren train satırları atıldı. Sonuçta sızıntı **%0**.
3. **Büyük harf girdiden çıkarılamaz.** Aynı Osmanlıca metin "Ateş" ve "ateş" gibi farklı yazımlarla eşleşiyor (8.190 durum). Değerlendirme hem orijinal haliyle hem de `turkish_norm` ile yapılmalı.
4. **Satırların %49'unda rakam var.** Rakamlar Arap-Hint rakamı (٣٨) olarak yazılmış ve Türkçe taraftakilerle birebir eşleşiyor (38). Gerçek içerik, artefakt değil.

## Uygulanan işlemler (`scripts/step2_clean_split.py`)

**Osmanlıca normalizasyon:** NFC; ي ve ى → ی; ک → ك; ە → ه; harekeler, tatweel ve görünmez karakterler (ZWNJ dahil) silindi; boşluklar tek boşluğa indirildi. Türkçe tarafta yalnızca NFC ve boşluk düzeltmesi yapıldı, **büyük harfler korundu.**

| Filtre | Atılan |
|---|---|
| > 512 UTF-8 byte (ByT5 sınırı) | 677 |
| Neredeyse sadece sayı ("٣١ ٣٨٥.") | 67.148 |
| İki taraftaki sayılar tutmuyor (hizalama hatası) | 65 |
| Uzunluk oranı < 0,5 veya > 3 | 1 |
| Tekrar | 35 |
| Sızıntı temizliği (train / dev) | 178.413 / 934 |

**Split:** Normalize edilmiş Osmanlıca metnin MD5 hash'ine göre %98 / %1 / %1. Deterministik: script tekrar çalıştırılırsa aynı split çıkar.

**Uzunluk:** Osmanlıca medyan 48 karakter (87 byte), p99 375 byte, en fazla 512 byte.

## Dış test setleri (`scripts/step3b_build_external.py`)

- Osmanlıca tarafa eğitimle **aynı** normalizasyon uygulandı. Ek olarak sunum biçimi harfler (ﺎ → ا) normal harflere çevrildi, ۔ noktaya çevrildi, eğitimde hiç geçmeyen noktalama (؟ ( ) « » " vb.) silindi. Eğitimde görülmemiş karakter kalmadı.
- Ot2Tr'de satır sayısı tutmayan 3 sayfa atıldı (103 sayfanın 100'ü kullanılıyor). 22 satır 512 byte'tan uzun; `ottoman_bytes` alanına bakılarak atlanabilir.
- **Türkçe referanslar bilimsel transliterasyonla yazılmış** ("Huzûr-ı Sâmî", "şer‘an"). `turkish_norm` alanı `scripts/text_norm.py` ile üretildi: küçük harf (Türkçe kuralları), â/î/û ve ayın/hemze işaretleri silinir, tire boşluğa çevrilir, noktalama silinir.
- **Kıyas noktası** (osmanlica.com'un kendi sistemi, `metrics.json` dosyalarından): Rika için transliterasyon CER 0,113, norm-CER 0,012; Nesih için CER 0,113, norm-CER 0,043. Onların "norm" tanımı bizimkiyle aynı olmayabilir, bu yüzden bu skorlar kaba referans olarak kullanılmalı.

## Değerlendirmede dikkat edilecekler

- Model çıktısı ile referans **aynı** `turkish_norm()` fonksiyonundan geçirilmeli. Raporlanacak skorlar: (a) ham CER/WER, (b) norm CER/WER.
- Dış setlerde CER'in iç teste göre yüksek çıkması beklenen bir durum. `text_norm` eski dil biçimlerini kapatmıyor: referans "etdirilmeyüp", "içün" yazarken model modern "ettirilmeyip", "için" üretecek. Bu fark ölçülmek istenen domain kaymasının bir parçası.
- Anlamlılık için önerim: test setleri üzerinde bootstrap güven aralığı (1.000 örneklem).

## Değerlendirme protokolü

**Setler** (`scripts/step4_make_eval_sets.py` → `data/eval/`): `test_2k` (iç test'ten seed 42 ile 2.000 satır), `riqa` (1.575), `naskh` (376), `ot2tr` (2.358). **Bütün modeller bu dosyalarla değerlendirilir.** `fewshot.jsonl` ise LLM'e gösterilen 8 örnek, train'den alındı.

**Tahmin formatı:** `preds/<model_adı>/<set>.jsonl`, her satır `{"id": "...", "prediction": "..."}`. id'ler `data/eval` dosyalarındakiyle aynı olmalı. Eksik tahmin boş string sayılır ve cezalandırılır.

**Metrikler** (`scripts/evaluate.py`, korpus seviyesinde, yüzde):
- **Ana metrik: norm-CER.** Tahmin ve referans `text_norm.turkish_norm()` fonksiyonundan geçirildikten sonra hesaplanan karakter hata oranı.
- Ek olarak ham CER, ham ve norm WER, chrF (sacrebleu).
- Her skor için %95 bootstrap güven aralığı (1.000 örneklem).

**"Gerçek fark" tanımı:** İki model arasında norm-CER farkı **≥ 1 puan** olmalı ve **paired bootstrap p < 0,05** çıkmalı. `--compare` seçeneği bunu otomatik raporluyor.

**Baseline'lar:**
| | Sistem | Açıklama |
|---|---|---|
| B0 | `baseline_rule.py` | Harf eşleme ve basit ünlü kuralları, öğrenme yok (alt sınır) |
| B1 | `baseline_llm_fewshot_colab.ipynb` | Qwen2.5-1.5B-Instruct, 8 örnekle eğitimsiz çeviri |
| B2 | osmanlica.com `metrics.json` | Yayınlanmış referans skorlar (Rika ve Nesih) |

**Komutlar:**
```bash
pip install rapidfuzz sacrebleu
python scripts/step4_make_eval_sets.py
python scripts/baseline_rule.py
python scripts/evaluate.py --model_dir preds/rule
python scripts/evaluate.py --model_dir preds/qwen_fewshot --compare preds/rule
```
Her model klasörüne `results.json` ve set başına en kötü 30 örneği içeren `worst_<set>.txt` yazılır. Hata analizi için bu dosyalar kullanılır.

## Tekrar üretmek

```bash
pip install datasets pandas pyarrow matplotlib
python scripts/step1_explore.py --out out/step1            # indir + istatistik
python scripts/step2_clean_split.py                         # normalizasyon, filtre, split
python scripts/step3a_peek_external.py                      # data/external/raw/*.zip aç
python scripts/step3b_build_external.py                     # dış setleri dönüştür
```
