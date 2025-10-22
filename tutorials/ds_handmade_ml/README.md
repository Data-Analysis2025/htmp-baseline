# 手づくりMNISTハンズオンガイド

このフォルダには、データサイエンス初学者が **90分前後** で Python と PyTorch を使った
機械学習パイプラインの「手作り体験」を行えるようにするための教材一式が含まれています。
本リニューアルでは、実務やコンペでよく見る `data/`・`models/`・`training/` の
責務分割を保ちながらも、必要最小限のファイル群にそぎ落として 90 分でも
学び切れるシンプルさを意識しました。学習進捗は `tqdm` のプログレスバーで
追跡できるため、手元での挙動を確認しながら安全に試行錯誤できます。

- [`mnist_hands_on.py`](./mnist_hands_on.py): サブコマンド方式の CLI エントリポイント。
  `inspect` と `train` でデータ理解から学習・評価までを切り替えて実行できます。
- `handmade_mnist/`
  - [`config.py`](./handmade_mnist/config.py): 乱数シードやデバイス設定を束ねたデータクラス。
  - [`data.py`](./handmade_mnist/data.py): DataLoader と前処理を構築し、形状確認やラベル集計を補助。
  - [`model.py`](./handmade_mnist/model.py): BatchNorm や Dropout を備えた軽量 CNN。
  - [`training.py`](./handmade_mnist/training.py): `tqdm` 付きの学習ループ、CSV ログ出力、評価・可視化ユーティリティ。

> **ゴール**: PyTorch と周辺ツールチェーン (Linux シェル、`screen`、`top`、`nvidia-smi` など) を
> 活用しながら、MNIST 画像分類モデルを一から構築・分析できるようになる。

## 0. 事前準備 (10分)

1. **リポジトリと Conda 環境のセットアップ**
   ```bash
   git clone <your-fork-url>
   cd htmp-baseline
   conda activate <既存の環境名>
   pip install -r requirements.txt
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install matplotlib tensorboard
   ```
   CPU で試す場合は上記の CUDA ホイールを CPU 版に差し替えてください。

2. **GPU/CPU リソースの確認**
   ```bash
   export CUDA_VISIBLE_DEVICES=1    # GPU を明示的に指定したい場合
   nvidia-smi                       # GPU の空き状況を確認
   watch -n 1 nvidia-smi            # 1秒ごとの変化を監視
   top                              # CPU/メモリの利用状況を監視
   htop                             # (インストール済みなら) より詳細な監視
   ```

3. **長時間ジョブのための端末分離**
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
python mnist_hands_on.py inspect --batch-size 32 --log-dir outputs/inspect
```

実行後は以下の操作を行ってみましょう。

- `p` コマンドでバッチ内テンソルを表示 (`p images[0, 0, :5, :5]` など)。
- `type(images)` や `images.numpy().dtype` を確認し、NumPy ↔ PyTorch 変換の違いを理解する。
- `dir(images)` や `images.shape`, `images.mean().item()` を使ってテンソル API を探る。
- `outputs/inspect/inspect_grid.png` を開き、正規化後のグレースケール画像がどのように見えるか確認。
- `outputs/inspect/label_distribution.csv` を `pandas.read_csv` で読み込み、ラベル偏りを可視化。

### 1-2. DataLoader の挙動を理解する

- `--limit-batches 5` を指定して反復回数を制御し、平均バッチサイズやラベル分布を観察。
- `--batch-size` や `--num-workers` の値を変更し、`top` で CPU 使用率の変化を観察。
- `--augment` を付与して Data Augmentation の効果を見比べる。
- `torch.manual_seed` の効果を試すため、`--seed` を複数値で比較。

## 2. モデル構築と学習ループ (40分)

### 2-1. 1エポックだけ走らせてみる

```bash
python mnist_hands_on.py train --epochs 1 --limit-batches 30 --log-dir outputs/run001
```

確認ポイント:

- 標準出力に表示される **tqdm プログレスバーとバッチ損失・精度** を読み解く。
- `screen` で別セッションを開き `nvidia-smi` を並行実行し、GPU 利用状況を監視。
- `top` と `watch -n 0.5 nvidia-smi` を並べて、I/O や GPU メモリの挙動を比較。
- `outputs/run001/epoch_metrics.csv` や `batch_metrics.csv` を `pandas` でロードし、再現性のあるロギング方法を理解。

### 2-2. `pdb` でテンソルを追跡

```bash
python mnist_hands_on.py train --limit-batches 2 --pdb --log-dir outputs/debug
```

- `n` / `s` / `c` を使って実行を進める。
- `p images.shape`, `p outputs.mean().item()` などで forward/backward 直前のテンソルを可視化。
- `p torch.cuda.memory_allocated()/1e6` で GPU メモリの変化を観察。

### 2-3. GPU と CPU の比較

```bash
python mnist_hands_on.py train --limit-batches 10 --device cuda --log-dir outputs/gpu
python mnist_hands_on.py train --limit-batches 10 --device cpu --log-dir outputs/cpu
```

- `time` コマンドで所要時間を比較。
- `nvidia-smi` と `top` でリソース利用を比較。
- `--batch-size 128` や `--augment` を組み合わせ、ボトルネック分析の練習。

### 2-4. 学習曲線と混同行列の可視化

```bash
python mnist_hands_on.py train --epochs 3 --plot-learning-curve --evaluate --log-dir outputs/full
```

- `outputs/full/learning_curve.png` で損失・精度の推移を確認。
- `outputs/full/confusion_matrix.csv` を開き、どの数字で誤分類が発生しているかを分析。

## 3. 仕上げと追加課題 (20分)

1. **Validation の挙動を理解する**: `--limit-batches -1` で全量評価し、学習率やバッチサイズを調整。
2. **モデル改造**: `handmade_mnist/model.py` に Dropout や BatchNorm の位置変更、
   あるいは Residual Block を追加して精度変化を検証。
3. **ログ取り**: `tensorboard --logdir outputs/` を起動し、学習曲線と指標の推移を横断比較。
4. **ユニットテスト**: `pytest` を利用して `training.accuracy` 関数の挙動をテストするコードを書いてみる。

## チートシート

| 作業内容 | 代表的なコマンド | メモ |
| --- | --- | --- |
| 仮想環境の有効化 | `conda activate <env>` | `which python` でパスを確認 |
| GPU 指定 | `export CUDA_VISIBLE_DEVICES=1` | Jupyter/Screen でも同じ変数を設定 |
| 画面分割 | `screen`, `tmux` | セッション名を付けると管理しやすい |
| GPU 監視 | `watch -n 1 nvidia-smi` | `persistence-mode` で初期化回避も検討 |
| CPU 監視 | `top`, `htop` | `1` キーで CPU コア表示を切り替え |
| Python デバッガ | `python -m pdb script.py` / `import pdb; pdb.set_trace()` | `l`/`n`/`s`/`c`/`p` が基本 |
| テンソル情報表示 | `tensor.shape`, `tensor.dtype`, `tensor.mean()` | `.item()` で Python スカラーに変換 |
| NumPy 変換 | `tensor.cpu().numpy()` | GPU テンソルは `.cpu()` が必須 |
| モデル保存 | `torch.save(model.state_dict(), "model.pt")` | CPU/GPU 変換に注意 |

## 推奨タイムテーブル

| 時間 (分) | 内容 |
| --- | --- |
| 0-10 | 環境準備、リソース確認、`screen` セッション作成 |
| 10-30 | データ探索 (`inspect` サブコマンド)、テンソル API に慣れる |
| 30-70 | 学習ループ実行、`pdb` を使ったデバッグ、GPU/CPU 比較 |
| 70-90 | 追加課題 (モデル改造、ログ取り、評価設定変更) |

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
