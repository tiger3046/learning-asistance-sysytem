# learning-asistance-sysytem
<h1>リポジトリの概要</h1>
本システムは、Raspberry Pi 及びRaspberry Piカメラを使用し撮影された利用者が学習しているかどうかを判定し、利用者のLINEに通知するシステムです。
以下はファイルの概要です。
learning assistance system.py は画像判定及びLINEによる通知を行うメインシステムです。
photo_collect.pyは、写真を収集します。learning assistance system.pyが正常に動くかどうかのチェックのために写真を集める際に使用してください。
oo.pyは写真が15枚に達したときに写真フォルダ内の写真をすべて消去します。

システムの目的
オンライン授業が増加し自己管理が従来の大学生よりもさらに必要になった大学生向けの学習支援システムです。
スマーフォンをいじり、課題に集中できていないなどの状況にLINEによる通知を行い、課題に取り組むように促します。

使い方
使用するために３つ必要なものがあります。1つ目は、RaspberryPi カメラとRaspberryPi等画像撮影及び15枚以上の画像集積が可能なツールです。本システムは、RaspberryPi カメラで1分間に一枚撮影し、その撮影した写真を画像判定するという前提で開発されたシステムです。ですので、RaspberryPi カメラ等画像を撮影しファイルに画像を集積するツールをご使用ください。2つ目は、2分類学習による機械学習済み.h5ファイルです。3つ目は、LINEnotifyのトークンです。LINEnotifyにログインいただき、ご自身に対して送信するトークンもしくは通知を希望されるLINEグループのトークンの取得を行ってください。上記の３つをご準備いただいたうえで下記の５つの箇所のコード書き換えが必要となります。1つ目は、57行目のline_notify_token = ''　の''にご自身の取得いただいたLINEトークンを入れてください。２つ目は、65行目のfile=""の""の中に.h5ファイルのパスを入れてください。３つ目は、66行目src_file = glob.glob('')の''に画像ファイルのパスを入れてください。4つ目は、86行目のsrc_file = glob.glob('')の''に画像ファイルのパスを入れてください。5つ目は、oo.pyファイルの6行目os.makedirs('')の''に画像ファイルのパスを入れてください。以上の必須ツール及び操作が完了していれば使用可能です。もし動かない場合は、下記の環境設定が完了しているかご確認下さい。確認方法は、windowsであれば、コマンドプロンプトにpython -version 及びpip list（もしくはpy -m pip list）と入力すれば、ご確認いただけます。

開発環境
pythonのバージョンは、python3.9.6を使用しています。また、tesorflow2.6.0、urllib1.26.6、Pillow8.3.1、numpy1.19.5、pip21.2.4、requests2.26.0を使用しています。
上記のものをインストールしたうえでエラーが起きる場合には、その都度エラーステイトメントを読み必要なツールをインストールしてください。

開発のポイント
約1分に1枚、RaspberryPiカメラによって撮られるので、15分間撮影し続け15枚の判定に基づき判定するようにできるようにすることにとても苦労しました。oo.pyを使用し、フォルダの写真をすべて消すことでsrc_file==15か否かで分別できるようにしました。また、システムがとても複雑でさまざまなライブラリを使用しています。
