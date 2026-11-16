# Janus – hangalapú deepfake-detektor

A Janus egy kis méretű detektor, amely egy beszédfelvételről eldönti, hogy valódi emberi beszéd vagy géppel (szövegfelolvasóval, hangátalakítóval) készült hamisítvány. A repó a szakdolgozatban vizsgált három változat teljes kódját tartalmazza, így a tanítás és a kiértékelés megismételhető.

> Császár Marcell (2026): *Deepfake- és hangalapú AI-támadások detektálása.* Szakdolgozat, Óbudai Egyetem, Keleti Károly Gazdasági Kar.

## Változatok

| Változat | Bemenet | Modell | Paraméter | Dev EER | Eval EER |
|---|---|---|---|---|---|
| Janus 0.1 | mel-spektrogram (80 × 251) | JanusCNN, 3 konvolúciós blokk | 23 585 | 8,31 ± 1,04% | 24,32 ± 3,48% |
| Janus 0.2 | LFCC + delta + delta-delta (60 × 401) | JanusCNN, 3 konvolúciós blokk | 23 585 | 11,02 ± 0,91% | 21,01 ± 1,69% |
| Janus 0.3 | LFCC + delta + delta-delta (60 × 401) | JanusResNet, 5 reziduális blokk | 1 224 257 | 4,84 ± 0,20% | 14,20 ± 0,39% |

Az értékek három tanítás (seed 0, 1, 2) átlaga és mintaszórása, az ASVspoof 2019 LA fejlesztő (dev) és teljes kiértékelő (eval) halmazán. A három változat ugyanazt a kódot, az előfeldolgozást és a tanítási beállításokat használja, csak a bemenet és a modell tér el. Részletes eredmények: [`results/summary.md`](results/summary.md).

## Felépítés

```
configs/   a három változat beállításai (bemenet, modell, augmentáció)
src/       features.py, dataset.py, model.py, train.py, evaluate.py, predict.py
scripts/   summarize_seeds.py (EER-összesítés), threshold_metrics.py (küszöb alapú metrikák)
results/   a 3 × 3 futás összesített eredményei (CSV és Markdown), logs/: a tanítási naplók
```

## Telepítés

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Adat

Az ASVspoof 2019 LA adathalmaz licenc miatt nem része a repónak. Letölthető innen: https://datashare.ed.ac.uk/handle/10283/3336

A `--data_root` a `LA` mappára mutasson, amelyben a `ASVspoof2019_LA_train`, `ASVspoof2019_LA_dev`, `ASVspoof2019_LA_eval` és `ASVspoof2019_LA_cm_protocols` mappák vannak.

## Tanítás

A szakdolgozat futásai ezekkel a beállításokkal készültek (Kaggle, NVIDIA T4):

```bash
python src/train.py --data_root data/LA --config configs/janus_03_lfcc.yaml \
    --epochs 30 --augment --sched cosine --batch 64 --lr 1e-3 --seed 0 \
    --cache_dir cache --out out/v03_seed0
```

A többi változathoz a `--config` értéke `configs/janus_01_mel.yaml` vagy `configs/janus_02_lfcc.yaml`. A tanítás minden kör végén a fejlesztő halmazon méri az EER-t, és a legjobb állapotot menti (`janus_best.pt`).

## Kiértékelés

```bash
python src/evaluate.py --data_root data/LA --checkpoint out/v03_seed0/janus_best.pt --split eval --out out/v03_seed0
python src/evaluate.py --data_root data/LA --checkpoint out/v03_seed0/janus_best.pt --split dev  --out out/v03_seed0
```

Kimenet: `eval_scores.csv` / `dev_scores.csv` (felvételenkénti pontszám) és `*_per_attack.csv` (EER támadásonként).

Összesítés több futásra:

```bash
python scripts/summarize_seeds.py --runs out --out results
python scripts/threshold_metrics.py --runs out --out results
```

A `threshold_metrics.py` a fejlesztő halmaz EER-küszöbét változtatás nélkül alkalmazza a kiértékelő halmazon, és ebből számol precisiont, recallt, F1-et, balanced accuracyt és támadásonként az átengedett hamis felvételek arányát.

## Egy hangfájl vizsgálata

```bash
python src/predict.py --checkpoint janus_03_seed0.pt --audio felvetel.wav --scores dev_scores.csv
```

A fájlt 4 másodperces darabokra vágja, mindegyiket pontozza, és az átlagot adja vissza. A `--scores` megadásakor a küszöb a fejlesztő halmaz EER-pontja. A detektor az ASVspoof 2019 LA adatain tanult, más forrású, tömörített vagy telefonon továbbított felvételeken az eredmény megbízhatatlan lehet.

## Betanított modellek

A 9 betanított modell (3 változat × 3 seed) a [Releases](../../releases) oldalon érhető el.

## Licenc

MIT, lásd [`LICENSE`](LICENSE).
