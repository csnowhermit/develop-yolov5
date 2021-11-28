# develop-yolov5

极简yolov5-手写版



## ISSUE

### 1、torch.load() 出现 No module named 'models'

​	原因：yolov5官方版在保存模型时候使用如下方式：

``` Python
if save:
    with open(results_file, 'r') as f:  # create checkpoint
        ckpt = {'epoch': epoch,
                'best_fitness': best_fitness,
                'training_results': f.read(),
                'model': ema.ema.module if hasattr(model, 'module') else ema.ema,
                'optimizer': None if final_epoch else optimizer.state_dict()}

    # Save last, best and delete
    torch.save(ckpt, last)
    if (best_fitness == fi) and not final_epoch:
        torch.save(ckpt, best)
    del ckpt
```

​	这种方式保存模型，会将模型源码的相对位置也默认保存（即模型中的models/目录和utils/目录）。所以使用torch.load()方式加载模型时也要保持这个目录结构，否则就会出现如上报错。

``` python
develop-yolov5
	|——models/
	|——utils/
	|——train.py等等
```

​	pytorch官方推荐保存模型的方式：

```python
torch.save(my_model.state_dict(), PATH)
```

​	按照pytorch官方文档，推荐的方式是将state_dict保存下来，相当于只保存训练结果中的权重和各种参数，这样加载时不会受到目录和class名等等的限制。

​	而yolov5官方直接保存模型的原因：yolov5中前向传播涉及到自适应锚框的计算（根据模型中保存的training_result自适应锚框计算，而不是提前定义好锚框），如果只保存权重参数，模型前向传播计算不了。

### 2、torch.nn.modules.module.ModuleAttributeError: 'Focus' object has no attribute 'ffrom'

​	出现该错误，主要原因是自己解析yolov5s.yaml等配置文件，构建网络时使用的模块名称与预训练模型中保存的模型结构不一致。

​	类似错误还有：

``` python
torch.nn.modules.module.ModuleAttributeError: 'Conv' object has no attribute 'acti'
torch.nn.modules.module.ModuleAttributeError: 'Bottleneck' object has no attribute 'conv1'
torch.nn.modules.module.ModuleAttributeError: 'Bottleneck' object has no attribute 'conv2'
torch.nn.modules.module.ModuleAttributeError: 'BottleneckCSP' object has no attribute 'conv3'
torch.nn.modules.module.ModuleAttributeError: 'BottleneckCSP' object has no attribute 'conv4'
torch.nn.modules.module.ModuleAttributeError: 'BottleneckCSP' object has no attribute 'module_list'
torch.nn.modules.module.ModuleAttributeError: 'Concat' object has no attribute 'dim'
torch.nn.modules.module.ModuleAttributeError: 'Detect' object has no attribute 'num_layers'
torch.nn.modules.module.ModuleAttributeError: 'Detect' object has no attribute 'num_anchors'
```

​	等等。

​	解决办法：推理时如果用预训练模型，则在构建网络结构时候使用与预训练模型相同的名字。



