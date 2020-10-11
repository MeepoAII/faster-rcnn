import torchvision

net = torchvision.models.vgg16()
print("Original features", net)
features = list(net.features[:30])
for feature in features:
    print(feature)