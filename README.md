# MedicalNer - 中文电子病历命名实体识别(pytorch实现)

### pytorch版本

- 1.4

### 数据集处理

- 解压 data_origin.zip, 运行 transfer_data.py,得到标注的数据集
- 按照自己需求获取train,test,dev三个数据集

### 修改配置config.yml

- 按照需要修改以下配置，tags为要识别的命名实体

        embedding_size: 300
        hidden_size: 128
        model_path: models/
        batch_size: 30
        dropout: 0.5
        tags:
          - TREATMENT
          - BODY
          - SIGNS
          - CHECK
          - DISEASE
### 训练

- 确保运行时有GPU,不然需要修改部分代码 (把torch.cuda()的所有.cuda() 删除)
- 通过数据集处理数据得到 train,test,dev 三个数据集
- 运行main.py

        1.python main.py
        2.输入 train 即可开始训练
        3.训练中每5个epoch评估一次模型,并记录在tensorboard中
        4.训练过程
        
        epoch [155] |████████████████████████████████████████████████████        | 26/30
            loss 88.828		last_loss 90.121		time 14.15625		best_avg_loss 116.693
        ------------------------------------------------------------------------------------------
        epoch [155] |██████████████████████████████████████████████████████      | 27/30
            loss 69.074		last_loss 69.539		time 13.78125		best_avg_loss 116.693
        ------------------------------------------------------------------------------------------
        epoch [155] |████████████████████████████████████████████████████████    | 28/30
            loss 59.598		last_loss 60.180		time 9.296875		best_avg_loss 116.693
        ------------------------------------------------------------------------------------------
        epoch [155] |██████████████████████████████████████████████████████████  | 29/30
            loss 65.137		last_loss 65.805		time 10.90625		best_avg_loss 116.693
        ------------------------------------------------------------------------------------------
        epoch [155] |████████████████████████████████████████████████████████████| 30/30
            loss 63.645		last_loss 64.129		time 9.65625		best_avg_loss 116.693
        ------------------------------------------------------------------------------------------
        epoch  155    avg_loss  116.0021484375    total_time  634.734375
            TREATMENT	origins:805.0			founds:806.0			rights:801.0
                    recall:0.9950310559006211	precision:0.9937965260545906	f1:0.9944134078212291
            BODY	origins:8278.0			founds:8292.0			rights:8101.0
                    recall:0.9786180236772167	precision:0.9769657501205982	f1:0.9777911888955945
            SIGNS	origins:6366.0			founds:6368.0			rights:6341.0
                    recall:0.9960728872133208	precision:0.9957600502512562	f1:0.995916444165227
            CHECK	origins:7718.0			founds:7738.0			rights:7688.0
                    recall:0.9961129826379891	precision:0.9935383820108555	f1:0.994824016563147
            DISEASE	origins:474.0			founds:473.0			rights:473.0
                    recall:0.9978902953586498	precision:1.0	f1:0.9989440337909187
            all_origins:23641.0			all_founds:23677.0			all_rights:23404.0
            all_recall:0.98997504335688	all_precision:0.9884698230350129	all_f1:0.9892218606027303

- 在tensorboard查看训练结果

        writer = SummaryWriter(log_dir='./tensorboard/5epoch')
        在main.py文件中修改log_dir的文件路径,在终端中进入log_dir路径下，输入以下命令:
        
        tensorboard --logdir=./
        
        点击 http://localhost:6006/ 
        即可查看各个实体的recall,found,f1,全部实体的recall,found,f1,以及每个epoch所有batch的平均loss
        
        
        
### 预测
- 运行main.py

        1.python main.py
        2.输入 predict 即可开始预测
        3.输入文本
        
### 评估
- 将要评估的数据集改名为 test
- 运行main.py

        1.python main.py
        2.输入 evaluate 即可开始评估




