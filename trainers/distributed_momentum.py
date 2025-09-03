import torch
from torch import nn
from .utils import test_model

class Client():
    def __init__(self, id, model, optimizer, trainset, batch_size, loss_fn, device, attack_fn=None, checkpointed_epoch=0):
        self.id = id
        self.device = device
        self.model = model
        self.optimizer = optimizer
        if attack_fn != None:
            self.dpa = attack_fn if type(attack_fn).__name__ == 'LabelFlip' else None
        else:
            self.dpa = None
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
        self.loss_fn = loss_fn
        self.data_iter = iter(self.trainloader)
        for _ in range(checkpointed_epoch % len(self.trainloader)):
            next(self.data_iter)

    def train(self):
        # print(next(self.model.parameters()).view(-1)[:10].data)
        try:
            images, labels = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.trainloader)
            images, labels = next(self.data_iter)
        images, labels = images.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(images)
        loss = self.loss_fn(output, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def load_state_dict(self, params, reset_num_batches_tracked=False):
        offset = 0
        state_dict = self.model.state_dict()

        for n, p in state_dict.items():
            if 'num_batches_tracked' not in n:
                size = p.numel()
                # The .copy_ is very important here
                p.copy_(params[offset:offset + size].view_as(p).detach().clone().to(dtype=p.dtype))
                offset += size
            elif reset_num_batches_tracked:
                p -= 1

        self.model.load_state_dict(state_dict)

    def get_state_dict_change(self, server_params):
        return torch.cat([p.view(-1) for n, p in self.model.state_dict().items() if 'num_batches_tracked' not in n]) - server_params

class Server():
    def __init__(self, model, testset, batch_size, device):
        self.model = model
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        self.device = device

    def load_state_dict(self, params):
        offset = 0
        state_dict = self.model.state_dict()

        for n, p in state_dict.items():
            if 'num_batches_tracked' not in n:
                size = p.numel()
                # The .copy_ is very important here
                p.copy_(params[offset:offset + size].view_as(p).detach().clone().to(dtype=p.dtype))
                offset += size
        
        self.model.load_state_dict(state_dict)
    
    def get_state_dict(self):
        return torch.cat([p.view(-1) for n, p in self.model.state_dict().items() if 'num_batches_tracked' not in n]) 

    def test(self):
        return test_model(self.model, self.testloader, self.device)
    

class DistributedMomentumTrainer():
    def init_actors(self, trainsets, testset, model, optimizer):
        loss_fn = nn.CrossEntropyLoss()
        server = Server(model, testset, self.exp.batch_size, self.device)
        clients = []
        for i in range(self.n_clients_to_train):
            clients.append(Client(i, model, optimizer, trainsets[i], self.exp.batch_size, loss_fn, self.device,
                                                attack_fn=None if i < self.num_clients - self.num_byz else self.attack_fn,
                                                checkpointed_epoch=self.exp.checkpointed_epoch))
        
        return server, clients
    
    def train_one_round(self):
        train_loss = 0
        self.updates = []
        server_params = self.server.get_state_dict()
        for i in range(self.n_clients_to_train):
            self.clients[i].load_state_dict(server_params, 
                                            reset_num_batches_tracked = True if i < self.n_clients_to_train - 1 else False)
            loss = self.clients[i].train()
            if i < self.num_clients - self.num_byz:
                train_loss += loss
            self.updates.append(self.clients[i].get_state_dict_change(server_params))

        train_loss /= (self.num_clients - self.num_byz)
        
        self.simulate_attack()

        self.after_attack_hook()

        server_params += self.agg_fn(self.updates)
        self.server.load_state_dict(server_params)
        return train_loss
