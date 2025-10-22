# 手づくりMNISTハンズオンガイド

このフォルダには、データサイエンス初学者が **90分前後** で Python と PyTorch を使った
機械学習パイプラインの「手作り体験」を行えるようにするための教材一式が含まれています。

- [`mnist_hands_on.py`](./mnist_hands_on.py): コマンドラインから実行できる学習・デバッグ用
  スクリプト。学習ループを短時間で再現でき、`.shape` や `dtype` の確認、`pdb` による
  デバッグ、GPU/CPU の切り替えなど、実務でも欠かせない操作を体験できます。
- 本 README: 90分の学習プラン、環境準備、操作チートシート、おすすめの追加課題をまとめた
  解説書。

> **ゴール**: PyTorch と周辺ツールチェーン (Linux シェル、`screen`、`top`、`nvidia-smi` など) を
> 活用しながら、MNIST 画像分類モデルを一から構築・観察できるようになる。

## 0. 事前準備 (10分)

1. **リポジトリのセットアップ**
   ```bash
   git clone <your-fork-url>
   cd htmp-baseline
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```
   GPU 環境を使用する場合は上記コマンドを適宜 CUDA 対応ホイールに置き換えてください。

2. **GPU/CPU リソースの確認**
   ```bash
   nvidia-smi            # GPU の空き状況を確認
   watch -n 1 nvidia-smi # 1秒ごとの変化を監視
   top                   # CPU/メモリの利用状況を監視
   htop                  # (インストール済みなら) より詳細な監視
   ```

3. **長時間コマンドのための端末分離**
   ```bash
   screen -S mnist-handson        # 新規セッション作成
   screen -ls                     # セッション一覧
   screen -r mnist-handson        # 再接続
   ctrl+a d                       # デタッチ
   ```

## 1. データ調査と基本操作 (20分)

### 1-1. MNIST の読み込みと形状・型チェック

```bash
cd tutorials/ds_handmade_ml
python mnist_hands_on.py --step inspect --batch-size 32
```

実行後は以下の操作を行ってみましょう。

- `p` コマンドでバッチ内テンソルを表示 (`p images[0, 0, :5, :5]` など)。
- `type(images)` や `images.numpy().dtype` を確認し、NumPy ↔ PyTorch 変換の違いを理解する。
- `dir(images)` や `images.shape`, `images.mean().item()` を使ってテンソル API を探る。

### 1-2. DataLoader の挙動を理解する

- `--limit-batches 5` を指定して反復回数を制御し、平均バッチサイズやラベル分布を観察。
- `batch_size` や `num_workers` の値を変更し、`top` で CPU 使用率の変化を観察。
- `torch.manual_seed` の効果を試すため、`--seed` を複数値で比較。

## 2. モデル構築と学習ループ (35分)

### 2-1. 1エポックだけ走らせてみる

```bash
python mnist_hands_on.py --step train --epochs 1 --limit-batches 30
```

確認ポイント:

- 標準出力に表示される **バッチ単位の損失と精度** を読み解く。
- `screen` で別セッションを開き `nvidia-smi` を並行実行し、GPU 利用状況を監視。
- もう一つの端末で `watch -n 0.5 python mnist_hands_on.py --step inspect` と比較実行し、
  I/O の挙動を観察。

### 2-2. `pdb` でテンソルを追跡

```bash
python mnist_hands_on.py --step train --limit-batches 2 --pdb
```

- `n` / `s` / `c` を使って実行を進める。
- `p images.shape`, `p outputs.mean().item()` などで forward/backward 直前のテンソルを可視化。
- `p torch.cuda.memory_allocated()/1e6` で GPU メモリの変化を観察。

### 2-3. GPU と CPU の比較

```bash
python mnist_hands_on.py --step train --limit-batches 10           # GPU 実行
python mnist_hands_on.py --step train --limit-batches 10 --cpu      # CPU 実行
```

- `time` コマンドで所要時間を比較。
- `nvidia-smi` と `top` でリソース利用を比較。
- `batch_size` を 128 などに変更し、メモリ使用量の違いを把握。

## 3. 仕上げと追加課題 (25分)

1. **Validation の挙動を理解する**: `ExperimentConfig.limit_batches` を `None` にして全量で
   評価し、学習率やバッチサイズを調整。
2. **モデル改造**: `SmallConvNet` に Dropout や BatchNorm を追加して精度変化を検証。
3. **ログ取り**: `tensorboard` を導入し、`SummaryWriter` を使って学習曲線を可視化。
4. **ユニットテスト**: `pytest` を利用して `accuracy` 関数をテストするコードを書いてみる。

## チートシート

| 作業内容 | 代表的なコマンド | メモ |
| --- | --- | --- |
| 仮想環境の有効化 | `source .venv/bin/activate` | Windows WSL の場合は `.venv\\Scripts\\activate` |
| 画面分割 | `screen`, `tmux` | セッション名を付けると管理しやすい |
| GPU 監視 | `watch -n 1 nvidia-smi` | `persistence-mode` で初期化回避も検討 |
| CPU 監視 | `top`, `htop` | `1` キーで CPU コア表示を切り替え |
| Python デバッガ | `python -m pdb script.py` / `import pdb; pdb.set_trace()` | `l`/`n`/`s`/`c`/`p` が基本 |
| テンソル情報表示 | `tensor.shape`, `tensor.dtype`, `tensor.mean()` | `.item()` で Python スカラーに変換 |
| NumPy 変換 | `tensor.numpy()` | GPU テンソルは `.cpu().numpy()` が必要 |
| モデル保存 | `torch.save(model.state_dict(), "model.pt")` | CPU/GPU 変換に注意 |

## 推奨タイムテーブル

| 時間 (分) | 内容 |
| --- | --- |
| 0-10 | 環境準備、リソース確認、`screen` セッション作成 |
| 10-30 | データ探索 (`inspect` ステップ)、テンソル API に慣れる |
| 30-65 | 学習ループ実行、`pdb` を使ったデバッグ、GPU/CPU 比較 |
| 65-90 | 追加課題 (モデル改造、ログ取り、評価設定変更) |

## トラブルシューティング

- **MNIST ダウンロードに失敗する**: プロキシ環境では `pip install torchvision` 時と同様に
  `HTTP_PROXY` / `HTTPS_PROXY` を設定。事前に `datasets.MNIST(..., download=True)` を別端末で
  実行してキャッシュしておくとスムーズです。
- **`nvidia-smi` が見つからない**: GPU ドライバがインストールされていない可能性があります。
  `which nvidia-smi` でパスを確認し、仮想環境内でも利用可能かチェックしてください。
- **`pdb` で GPU テンソルを表示できない**: `tensor.device` を確認し、`tensor.cpu().numpy()` を
  使用して CPU に移動してから NumPy 変換する必要があります。
- **`screen` が使いにくい**: `tmux` や VS Code のターミナル分割など、代替ツールも試してみて
  ください。

## 次のステップ

このハンズオンを終えたら、以下の方向に発展させてみてください。

- **データ拡張 (Data Augmentation)**: `transforms.RandomRotation` などを追加して精度向上を狙う。
- **学習曲線の可視化**: `matplotlib` や `seaborn` で損失・精度の推移を描画するスクリプトを作る。
- **推論 API 化**: `FastAPI` や `Flask` を用いて学習済みモデルを Web API として公開する。
- **MLOps 体験**: Dockerfile を書いて再現可能な実行環境を整備し、`docker stats` でリソース監視を行う。

楽しみながら、現場で求められる一連の流れを体験してみてください！
