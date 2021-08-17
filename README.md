# learning-asistance-sysytem
オンライン授業が増加し自己管理が従来の大学生よりもさらに必要になった大学生向けの学習支援システムです。
ラズベリーパイカメラで一分間に撮影した写真を２分類学習より機械学習したファイルによる画像判定を行い画像が勉強している風の画像か否かを判定します。
その後その他判定に基づき利用者にLINEでの通知を行います。
以下はファイルそれぞれの説明です。
learning assistance system.py は画像判定及びLINEによる通知を行うメインシステムです。
photo_collect.pyは、写真を収集します。learning assistance system.pyが正常に動くかどうかのチェックのために写真を集める際に使用してください。
oo.pyは写真が15枚に達したときに写真フォルダ内の写真をすべて消去します。
