## README

#### 模型训练

##### 指令

```bash
python train.py
```

##### 训练模型保存位置

```
model/model.pkl
```

---

#### 单条数据预测dfs

##### 输入

`test.yaml`文件中修改以下参数

```yaml
new_treat: 0
new_chem: 0
new_rad: 0
new_imu: 0
new_cnt: 0
post_treat: 1
post_chem: 1
post_rad: 1
post_target: 0
post_imu: 0
post_cnt: 2
progress: 1
death: 0
re_treat: 1
re_chem: 0
re_rad: 0
re_ope: 0
re_target: 1
re_imu: 0
re_cnt: 0
```

##### 指令

```
python test.py
```

##### 输出

DFS预测结果：

```
[329.03266667]
```

---

Contribution: Code completed by Bai Zhixin.
