# korean-conversation
한국어 대화를 진행하는 인공지능을 개발하고자 시도 중인 취미 프로젝트입니다.

### 21.10.10
모든 어절을 vocab.txt에 저장해두었습니다.    
숫자는 ${number}/SN 로 토크나이징 될 것입니다.

형태소를 뽑아내는 데에 KoNLPy Mecab 클래스를 사용하였습니다.    
mecab 클래스를 약간 수정하여 "+" 된 어절을 전부 분리하도록 하였습니다.    
[mecab/mecab_tokenizer.py](https://github.com/HanGyeolee/korean-conversation/blob/main/mecab/mecab_tokenizer.py)를 통해 분리된 형태소들을 토크나이징할 수 있습니다.

vocab.txt에 들어간 전체 어휘의 개수는 816287개 입니다.

<b>Ex)</b>    
> 경기 성남시 판교신도시에서 이달 분양하는 중대형 아파트의 3.3m²당 분양가가 2006년보다 200만 원 정도 싼 1500만 원 후반대로 결정될 것으로 보인다.

``` json
[
  218296, 724373, 317499, 396668, 812437, 254248, 191715, 350259, 249027, 291460, 
  816276, 4587, 372880, 816256, 10582, 191764, 816287, 816288, 816287, -1, 
  -1, 816164, 291460, 816137, 191481, 816287, 207053, 191664, 816287, 418926, 
  207370, 365195, 767213, 816284, 816287, 418926, 207370, 413800, 191548, 217693, 
  816266, 816285, 206811, 191746, 771756, 816286, 765791
]
```
