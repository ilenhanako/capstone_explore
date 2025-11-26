data flow
1) pretrain with online
2) eval with golden data
3) inference with pretrained model, 2 annotators correct it, creating a reliable annotated dataset
3) finetune with CAG
4) eval with golden data


with active learning will be
1) pretrain with online
2) eval with golden data
3) pick out top k images with pretrained model, conduct inference and 
 2 annotators correct it, creating a reliable annotated dataset
3) finetune with CAG
4) eval with golden data