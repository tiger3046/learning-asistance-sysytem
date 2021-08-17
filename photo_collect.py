from icrawler.builtin import BingImageCrawler

# 猫の画像を100枚取得
crawler = BingImageCrawler(storage={"root_dir": "nisino"})
crawler.crawl(keyword="勉強している人", max_num=15)
