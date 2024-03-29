# learning-asistance-sysytem

<h3>リポジトリの概要</h3>
本システムは、Raspberry Pi 及びRaspberry Piカメラを使用し撮影された利用者が学習しているかどうかを判定し、利用者のLINEに通知するシステムです。
<br>以下はファイルの概要です。
learning assistance system.py は画像判定及びLINEによる通知を行うメインシステムです。
photo_collect.pyは、写真を収集します。learning assistance system.pyが正常に動くかどうかのチェックのために写真を集める際に使用してください。
oo.pyは写真が15枚に達したときに写真フォルダ内の写真をすべて消去します。machine_learning1.pyは画像の機械学習を行います。

<h3>システムの目的</h3>
オンライン授業が増加し,従来の大学生よりもさらに学習に集中して向かうサポートが必要になった大学生向けの学習支援システムです。
スマーフォンをいじり、課題に集中できていないなどの状況にLINEによる通知を行い、課題に取り組むように促します。

<h3>使い方</h3>
使用するために３つ必要なものがあります。<br>⓵RaspberryPi カメラとRaspberryPi等画像撮影及び15枚以上の画像集積が可能なツールです。本システムは、RaspberryPi カメラで1分間に一枚撮影し、その撮影した写真を画像判定するという前提で開発されたシステムです。ですので、RaspberryPi カメラ等画像を撮影しファイルに画像を集積するツールをご使用ください。<br>⓶2分類学習による機械学習済み.h5ファイルです。こちらは、勉強している風画像と勉強していない風画像をそれぞれ大量にご準備いただく必要があります。machine_learning1.pyで機械学習を行うことが可能です。その際には、本システム内の学習ファイルのパスの設定の変更をしてください。私たちは約8000枚の画像を用いて機械学習を行いましたが、精度は30％というところでした。そのため、高精度のシステムを開発したいという方は本ファイルの書き換えと数万枚以上の学習用画像をご用意ください。<br>⓷LINEnotifyのトークンです。LINEnotifyにログインいただき、ご自身に対して送信するトークンもしくは通知を希望されるLINEグループのトークンの取得を行ってください。<br>上記の３つをご準備いただいたうえで下記の５つの箇所のコード書き換えが必要となります。<br>⑴は、57行目のline_notify_token = ''　の''にご自身の取得いただいたLINEトークンを入れてください。<br>⑵65行目のfile=""の""の中に.h5ファイルのパスを入れてください。<br>⑶66行目src_file = glob.glob('')の''に画像ファイルのパスを入れてください。<br>⑷86行目のsrc_file = glob.glob('')の''に画像ファイルのパスを入れてください。<br>⑸oo.pyファイルの6行目os.makedirs('')の''に画像ファイルのパスを入れてください。<br>以上の必須ツール及び操作が完了していれば使用可能です。もし動かない場合は、下記の環境設定が完了しているかご確認下さい。確認方法は、windowsであれば、コマンドプロンプトにpython -version 及びpip list（もしくはpy -m pip list）と入力すれば、ご確認いただけます。

<h3>開発環境</h3>
pythonのバージョンは、python3.9.6を使用しています。また、tesorflow2.6.0、urllib1.26.6、Pillow8.3.1、numpy1.19.5、pip21.2.4、requests2.26.0を使用しています。
上記のものをインストールしたうえでエラーが起きる場合には、その都度エラーステイトメントを読み必要なツールをインストールしてください。

<h3>開発のポイント</h3>
約1分に1枚、RaspberryPiカメラによって撮られるので、15分間撮影し続け15枚の判定に基づき判定するようにできるようにすることにとても苦労しました。oo.pyを使用し、フォルダの写真をすべて消すことでsrc_file==15か否かで分別できるようにしました。また、システムがとても複雑でさまざまなライブラリを使用しています。また、スクレイピングで機械学習用画像を収集しようと試みましたがあまり集まらなかったため、Raspberry Piとそのカメラモジュールで学習している風画像と学習していない風画像約8000枚を手動で撮影し機械学習を行いました。今後、 mysql.connectorにより取得したデータをmysqlによるデータベースに蓄積、及び蓄積したデータをopenpyxlによりデータをエクセルファイルにプロットなどを考えております。

<h3>開発担当箇所に関して</h3>
私は、主にlearning assistance system.pyの設計とコーディングを行いました。また、機械学習用画像の撮影も行っております。

<h3>動画に関して</h3>
システムのデモ動画です。システム開発の参考にしていただけますと、幸いです。

<h3>その他</h3>
本システムは、群馬プログラミングアワードにてTask Assistance Gunmatyanとして発表しており、群馬プログラミングアワードで検索いただくとプレゼン発表や音声通知のデモがご覧いただけます。チーム名は、まえこくlaboです。宜しければそちらも合わせてご覧下さい。
