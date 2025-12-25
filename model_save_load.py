import torch, torchvision
vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1       --> 自定义模型的类定义需要放在同一个文件中（小陷阱）
# torch.save(vgg16, 'vgg16_method1.pth')

# 保存方式2      【推荐】
# torch.save(vgg16.state_dict(), 'vgg16_method2.pth')

# 加载方式1
# vgg16 = torch.load('vgg16_method1.pth')

# 加载方式2
# vgg16 = torchvision.models.vgg16(pretrained=False)
# vgg16.load_state_dict(torch.load('vgg16_method2.pth'))

print(vgg16)