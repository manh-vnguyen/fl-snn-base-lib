import torch
from torch import nn

def test_model(model, testloader, device):
    with torch.no_grad():
        acc, total = 0, 0 
        for images, labels in iter(testloader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            acc += torch.sum(torch.argmax(output, dim=1) == labels)
            total += len(labels)
    
    return (acc/total).item()

### MOMENTUM SETTINGS
class ClientGlobalMomentum():
    def __init__(self, id, model, trainset, batch_size, loss_fn, device, attack_fn=None, starting_epoch=0, shuff_gen=None):
        self.id = id
        self.device = device
        self.model = model
        if attack_fn != None:
            self.dpa = attack_fn if type(attack_fn).__name__ == 'LabelFlip' else None
        else:
            self.dpa = None
        self.trainloader = torch.utils.data.DataLoader(trainset, 
                                                       batch_size=batch_size, 
                                                       shuffle=True, 
                                                       generator=shuff_gen)
        self.loss_fn = loss_fn
        self.data_iter = iter(self.trainloader)
        for _ in range(starting_epoch % len(self.trainloader)):
            next(self.data_iter)

    def train(self):
        try:
            images, labels = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.trainloader)
            images, labels = next(self.data_iter)
        images, labels = images.to(self.device), labels.to(self.device)
        if self.dpa != None:
            images, labels = self.dpa(images, labels)
        self.model.zero_grad()
        output = self.model(images)
        loss = self.loss_fn(output, labels)
        loss.backward()
        return loss.item()

    def get_model_grad(self):
        return torch.cat([p.grad.view(-1) for p in self.model.parameters()])


class ServerGlobalMomentum():
    def __init__(self, model, optimizer, testset, batch_size, device):
        self.model = model
        self.optimizer = optimizer
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        self.device = device
    
    def load_model_grad(self, grads):
        offset = 0
        for p in self.model.parameters():
            p.grad = grads[offset : offset + p.numel()].view_as(p).detach().clone()
            offset += p.numel()

    def step(self):
        self.optimizer.step()

    def test(self):
        return test_model(self.model, self.testloader, self.device)

def init_actors_global_momentum(args, model, optimizer, trainsets, testset, attack_fn, shuff_gen):
    server = ServerGlobalMomentum(model, optimizer, testset, args.batch_size, args.device)
    clients = []
    for i in range(args.num_clients):
        if i < args.num_clients - args.num_byz:
            clients.append(ClientGlobalMomentum(i, model, trainsets[i], args.batch_size, nn.CrossEntropyLoss(), args.device, 
                                                attack_fn=None,
                                                starting_epoch=args.checkpointed_epoch + 1,
                                                shuff_gen=shuff_gen))
        else:
            clients.append(ClientGlobalMomentum(i, model, trainsets[i], args.batch_size, nn.CrossEntropyLoss(), args.device, 
                                                attack_fn=attack_fn,
                                                starting_epoch=args.checkpointed_epoch + 1,
                                                shuff_gen=shuff_gen))
    return server, clients

def train_one_round_global_momentum(self):
    train_loss = 0
    updates = []
    for i in range(self.n_clients_to_train):
        loss = self.clients[i].train()
        if i < self.num_clients - self.num_byz:
            train_loss += loss
        updates.append(self.clients[i].get_model_grad())
    train_loss /= (self.num_clients - self.num_byz)

    if self.attack_fn != None:
        if type(self.attack_fn).__name__ == 'LabelFlip':
            pass
        elif type(self.attack_fn).__name__ == 'SignFlip':
            for i in range(self.num_clients - self.num_byz, self.num_clients):
                updates[i] *= -1
        elif type(self.attack_fn).__name__ == 'GaussRandom':
            m_update = self.attack_fn(updates[0].shape)
            updates += [m_update for _ in range(self.num_byz)]
        else:
            m_update = self.attack_fn(updates)
            updates += [m_update for _ in range(self.num_byz)]

    agg_grad = self.agg_fn(updates)
    self.server.load_model_grad(agg_grad)
    self.server.step()
    return train_loss
    