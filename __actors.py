import torch
from torch import nn
### MOMENTUM SETTINGS
def local_momentum_classes_and_functions():
    class Client():
        def __init__(self, id, model, optimizer, trainset, batch_size, loss_fn, device):
            self.id = id
            self.device = device
            self.model = model
            self.optimizer = optimizer
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
            self.loss_fn = loss_fn
            self.data_iter = iter(self.trainloader)

        def train(self):
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
                    p = params[offset:offset + size].view_as(p).detach().clone().to(dtype=p.dtype)
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
                    p = params[offset:offset + size].view_as(p).detach().clone().to(dtype=p.dtype)
                    offset += size
            
            self.model.load_state_dict(state_dict)
        
        def get_state_dict(self):
            return torch.cat([p.view(-1) for n, p in self.model.state_dict().items() if 'num_batches_tracked' not in n]) 

        def test(self):
            with torch.no_grad():
                acc, total = 0, 0 
                for images, labels in iter(self.testloader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    output = self.model(images)
                    acc += torch.sum(torch.argmax(output, dim=1) == labels)
                    total += len(labels)
            
            return (acc/total).item()
    
    def init_actors(args, model, optimizer, trainsets, testset):
        server = Server(model, testset, args.batch_size, args.device)
        clients = []
        for i in range(args.num_clients):
            clients.append(Client(i, model, optimizer, trainsets[i], args.batch_size, nn.CrossEntropyLoss(), args.device))
        return server, clients

    def train_one_round(self):
        train_loss = 0
        changes = []
        server_params = self.server.get_state_dict()
        for i in range(self.n_benign_clients):
            if i < self.n_benign_clients - 1:
                self.clients[i].load_state_dict(server_params, reset_num_batches_tracked=True)
            else:
                self.clients[i].load_state_dict(server_params, reset_num_batches_tracked=False)
            train_loss += self.clients[i].train()
            changes.append(self.clients[i].get_state_dict_change(server_params))
        train_loss /= self.n_benign_clients
        if self.attack_fn != None:
            m_grad = self.attack_fn(changes)
            changes += [m_grad for _ in range(self.num_byz)]
        server_params += self.agg_fn(changes)
        self.server.load_state_dict(server_params)
        return train_loss
    
    return init_actors, train_one_round

def global_momentum_classes_and_functions():
    class Client():
        def __init__(self, id, model, trainset, batch_size, loss_fn, device):
            self.id = id
            self.device = device
            self.model = model
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
            self.loss_fn = loss_fn
            self.data_iter = iter(self.trainloader)

        def train(self):
            try:
                images, labels = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.trainloader)
                images, labels = next(self.data_iter)
            images, labels = images.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            output = self.model(images)
            loss = self.loss_fn(output, labels)
            loss.backward()
            return loss.item()

        def get_model_grad(self):
            return torch.cat([p.grad.view(-1) for p in self.model.parameters()])


    class Server():
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
            with torch.no_grad():
                acc, total = 0, 0 
                for images, labels in iter(self.testloader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    output = self.model(images)
                    acc += torch.sum(torch.argmax(output, dim=1) == labels)
                    total += len(labels)
            
            return (acc/total).item()
 
    def init_actors(args, model, optimizer, trainsets, testset):
        server = Server(model, optimizer, testset, args.batch_size, args.device)
        clients = []
        for i in range(args.num_clients):
            clients.append(Client(i, model, trainsets[i], args.batch_size, nn.CrossEntropyLoss(), args.device))
        return server, clients
  
    def train_one_round(self):
        train_loss = 0
        grads = []
        for i in range(self.n_benign_clients):
            train_loss += self.clients[i].train()
            grads.append(self.clients[i].get_model_grad())
        train_loss /= self.n_benign_clients
        if self.attack_fn != None:
            m_grad = self.attack_fn(grads)
            grads += [m_grad for _ in range(self.num_byz)]
        agg_grad = self.agg_fn(grads)
        self.server.load_model_grad(agg_grad)
        self.server.step()
        return train_loss
    return init_actors, train_one_round
    