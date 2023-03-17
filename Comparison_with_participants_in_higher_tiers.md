## 1st place
* [Solution Overview](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394812)
* code 公開予定あり
* Major difference from my solution
    * 1st_stage_modelのtitle textのcontextですべてを使うのではなく系統上3つを使っている.
    * 2nd_stage_modelで使った特徴量で面白そうなもの
      * tf-idf(英語だけではないから厳しいかなと思ったけれど考え直したら使えたな)
    * transformerの訓練にArcFaceを使っている
      * ArcFaceは例えばMNISTのようにラベルの種類が決まっているものだけでしか使えないと勝手に思い込んでいたけれどtopicやcontentの方をラベルとして扱うことで使えるっぽい

## 2nd place
* [Solution Overview](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/395110)
* code 公開予定あり
* Major difference from my solution
    * 2nd_stage_modelを用いていない
    * 入力でtokenを新しく設定する代わりに#を用いている
    * symmetric contrastive loss-functionとしてInfoNCE Lossをつかっている(NCE stands for Noise-Contrastive Estimation)
       * symmetricって結構使えるよね. 自分が使ったMNSLossでも用いているし、調べてみたらArcFaceでもありそう-> Chatgptでcode出してもらった
    * 英語翻訳(epochごとの切り替え)
    * 知識蒸留
    * 量子化 

## 3rd place
* [Solution Overview](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394838)
* [code](https://github.com/syzong/2023-Kaggle-LECR-Top3-TrainCode)
* Major difference from my solution
    * 1st_stage_modelのloss function:  unsupervised SIMCSE 
        * これはDiscussionで話題になってたので試してみたけれど上手くいかなかった記憶がある。しかし挙げられているようなcodeは書いた覚えがないので私の実装が間違ってたんだろうな
    * 2nd_stage_modelのmodel
        * mdeberta: これは1st_Stage_modelの訓練の段階で時間がかなりかかりそうで(A100で10時間超え)かつbatch_sizeが大きく取れなかったのでMNSLossには不向きだと判断し断念した。コード見た感じ損失関数部分以外に変わりはなかったので損失関数かGPUパワーの違いかな -> mdeberta about 3-4 days on A100 because fp16 not working on mdeberta, xlmr base about 1 day and large about 2-3 days. パワーでした。
        * 多様性のために調整済みモデルなしでfinetuningを始めたものと調整済みモデルでfinetuningを混ぜている
        * 敵対的使っているが、時間かかるしその領域まで私はたどり着いていないのでまだ封印.
        


## 4th place
* Solution Overview(https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394984)
* Major difference from my solution
    * 1st_stage_modelと後処理のみ
    * 1st_stage_modelの訓練にArcFaceを使用、シンメトリーっぽい. code欲しいな


## 5th place
* Solution Overview(https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394827)
* Major difference from my solution
    * LB probing. そんな技が.
    * 1st_stage_model loss MegaBatchMarginLoss: 何でこれを使うっていう発想にならなかったんだろう。MNSLossに引っ張られすぎたな
    * 2nd_stage_modelにLightGBMを使用.

## 6th place
* Solution Overview(https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394813)
* Major difference from my solution
    * ArcFaceを使用
    * 2nd_stage_modelにLightGBMとCatBoostを使用


## 7th place
* Solution Overview(https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394946)
* Major difference from my solution
    * MLM + SimCSE ベースの 3 ラウンドの教師あり対照的な事前トレーニング。
        * MLMを試そうとは思わなかったな. MLMで精度を挙げられたことがないので諦めていた. あとSimCSEは強いみたい.
