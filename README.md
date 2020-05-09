#实现copy了论文.



#**GAT**实现
主要在gat_layer.py和gat.py中

gat_layers:这几行代码实现了主要的3个步骤

 ``` python
h_prime = torch.matmul(h.unsqueeze(0), self.w)                                     
attn_src = torch.bmm(h_prime, self.a_src)                                          
attn_dst = torch.bmm(h_prime, self.a_dst)                                          
attn = attn_src.expand(-1, -1, n) + attn_dst.expand(-1, -1, n).permute(0, 2, 1)    
attn = self.leaky_relu(attn)      
attn = self.softmax(attn)
```

![Image text](./1.png)

gat:串联用于中间层，将平均值用于最后一层

![Image text](./2.png)

#**GCN实现**

重点2行代码
 ``` python
support = torch.bmm(x, expand_weight)
output = torch.bmm(lap, support)
 ``` 
![Image text](./3.png)

**实现了论文中的AXW**


(好简单..分析完感觉又能重零写系列)




