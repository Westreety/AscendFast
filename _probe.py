import torch, torch_npu
print('torch', torch.__version__)
print('has compile', hasattr(torch, 'compile'))
try:
    import torchair
    print('torchair module:', torchair.__file__)
    from torchair import get_npu_backend
    print('get_npu_backend OK')
except Exception as e:
    print('torchair NO:', repr(e))
