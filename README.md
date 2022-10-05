# BCI release description

### Code

**EEGNet code**

- https://github.com/AilurusUmbra/EEGNet-tutorial/blob/master/EEGNet.py
- 여기 참고하시면 되겠습니다.

**Modifed-CRAM code**

- 첨부파일에 있는 코드는 모두 Modified-CRAM에 관련한 코드입니다.
- BCI2021이 Modified-CRAM와 동일한 모델입니다.
- EEGNet은 위에 있는 코드랑 논문 보시면서 모델 구현하시고, BCI2021와 관련된 코드에서 모델 부분만 추가하시면 될 것 같습니다.



### Modified-CRAM Usage

```bash
main.py --seed 42 --labels=0,1 --net=BCI2021 --band=0,42 --val_subject=1 --segment --sch=exp --gamma=0.993 --stamp=for_test
```

**argparse 설명**

- seed: 랜덤 조절하는 변수입니다. 42로 고정시키고 하시면 될 것 같아요.
- labels: 사용할 class 결정하는 변수입니다. 0, 1로 하시면 왼손 오른손 사용하는거고, 그대로 하시면 돼요.
- net: 어떤 모델을 사용할지 결정하는 변수입니다. BCI2021로 하시면 Modified-CRAM 모델 사용하는거에요.
- band: bandpass 대역폭 정하는 변수입니다. 시작은 0, 42로 하시고 적당히 조절하시면 될 것 같아요.
- segment: 데이터에 segmentation 적용하는 변수입니다.
- sch: learning scheduler 변수입니다. Exp는 exponential learning scheduler 사용하는거에요. 다른 스케줄러도 있으니까 코드 보시고 사용하시거나 추가하시면 돼요.
- gamma: Exponential learning scheduler에 사용되는 인자입니다. 다른 스케줄러 사용하실 때는 다른 인자를 추가하시는 것을 추천드려요.
- stamp: 결과 저장할 폴더 이름입니다. for_test로 하게 되면, "./result/for_test/0" 여기에 결과가 저장돼요. 다음 실험에서도 for_test를 사용하시면 결과가 "./result/for_test/1"에 저장돼요. 자신이 정한 디렉토리 안에 0번부터 넘버링해서 차례대로 저장하게 짰어요.
- 이것 말고도 argparse가 많아서 직접 보시는게 더 좋을 것 같아요. 이해 안가시는 부분은 모두 물어봐주세요!



### 추가 사항

- Evaluation 관련해서, "성능은 testset의 각 subject의 accuracy로 평가함. Trainset에 대해서 5-fold cross validation으로 model selection 하는 것을 추천"이라고 했었는데 이 부분에 대한 코드는 없어요. 직접 작성하시면 될 듯 해요.
- 지금 드린 코드는 session 1으로 train하고, session2로 validation/test하는 코드에요. 사실상 testset이 없는 코드죠. 빠른 실험을 위한 코드였어요. 정리하자면, sessions1 데이터로 validation까지 하고, session2 데이터는 test만 진행하시면 돼요.