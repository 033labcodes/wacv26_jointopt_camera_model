CSS
こちらのデータベースの，先行研究で求められたCSSを使用させていただいています．
HSIと合うように，線形補完しています．

使用する際は，以下の論文を引用してください．
Jun Jiang, Dengyu Liu, Jinwei Gu, and Sabine S¨usstrunk. What is the space of spectral sensitivity functions for digital color cameras? In IEEE Winter Conf. Appl. Comput. Vis.,pages 168–179, 2013. 6

CCM
カラーチェッカーのスペクトルをつかって，sRGB値と一致するようにカメラごとに事前に求めている．d65光源を使用している．
he BabelColor Company. Color Checker Pages, 2004. Last accessed on June 9, 2025. 4
CIE 2019, CIE standard illuminant D65, International Commission on Illumination (CIE), Vienna, AT, DOI: 10.25039/CIE.DS.hjfjmt59

XYZ CMF
ICE XYZ CMFを使用しています．
事前にHSIと合うようにダウンサンプリングしています．
CIE 2019, Colour-matching functions of CIE 1931 standard colorimetric observer, International Commission on Illumination (CIE), Vienna, AT, DOI: 10.25039/CIE.DS.xvudnb9b