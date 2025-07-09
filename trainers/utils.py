import torch

def test_model(model, testloader, device):
    with torch.no_grad():
        acc, total = 0, 0 
        for images, labels in iter(testloader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            acc += torch.sum(torch.argmax(output, dim=1) == labels)
            total += len(labels)
    
    return (acc/total).item()