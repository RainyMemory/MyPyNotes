## How to use cuda in pytorch
<br>

### When loading dataset
- Need to label your data to be load in the cuda() mode
```python
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255.   # Tensor on GPU
    test_y = test_data.test_labels[:2000].cuda()
```
---

### When running the module
- Make sure you reach your data batch from the gpu
```python
    batch_x = x.cuda()    # Tensor on GPU
    batch_y = y.cuda()    # Tensor on GPU
```
- Make sure your parameters are sent to the gpu to compute
```python
    test_output = cnn(test_x)
    pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU
```
---

### At testing time
- Same as before, do this in your gpu
```python
    pred_y = torch.max(test_output, 1)[1].cuda().data # move the computation in GPU
```
---

### Hint
- If you want to move your computation back to the cpu then you go:
```python
    pred_y = pred_y.cpu()
```