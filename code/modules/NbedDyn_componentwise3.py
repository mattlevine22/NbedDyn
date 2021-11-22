from code import *
import numpy as np
import torch
from pdb import set_trace as bp
from plotting_utils import *

def get_NbedDyn_model(params, use_f0=True, doMark=True, doNonMark=False):
        np.random.seed(params['seed'])
        torch.manual_seed(params['seed'])
        class FC_net(torch.nn.Module):
                    def __init__(self, params):
                        super(FC_net, self).__init__()
                        self.D = params['dim_latent']
                        self.K = 9
                        self.use_f0 = use_f0
                        self.doMark = doMark
                        self.doNonMark = doNonMark
                        self.f0only = self.use_f0 and not(self.doMark) and not(self.doNonMark)
                        y_aug = np.random.uniform(size=(params['nb_batch'],params['Batch_size'],self.K*params['dim_latent']))-0.5
                        for k in range(self.K):
                            y_aug[:,:, k*self.D+1 : (k+1)*self.D] = 0.0 # only have non-zeros at first dimension of each latent component
                        self.y_aug = torch.nn.Parameter(torch.from_numpy(y_aug).float())

                        if self.doNonMark:
                            self.set_hidden_NN()
                            self.set_output_NN()

                        if self.doMark:
                            self.set_Markovian_NN()

                        if self.use_f0:
                            self.f0 = params['f0']

                    def set_hidden_NN(self):
                        self.linearCell   = torch.nn.Linear(params['dim_latent']+params['dim_observations'], params['dim_hidden_linear'])
                        self.BlinearCell1 = torch.nn.ModuleList([torch.nn.Linear(params['dim_latent']+params['dim_observations'], 1,bias = False) for i in range(params['bi_linear_layers'])])
                        self.BlinearCell2 = torch.nn.ModuleList([torch.nn.Linear(params['dim_latent']+params['dim_observations'], 1,bias = False) for i in range(params['bi_linear_layers'])])
                        augmented_size    = params['bi_linear_layers'] + params['dim_hidden_linear']
                        self.transLayers = torch.nn.ModuleList([torch.nn.Linear(augmented_size, params['dim_latent'])])
                        self.transLayers.extend([torch.nn.Linear(params['dim_latent'], params['dim_latent']) for i in range(1, params['transition_layers'])])

                    def set_output_NN(self):
                        self.linearCell_out   = torch.nn.Linear(params['dim_latent'], params['dim_hidden_linear'])
                        self.BlinearCell1_out = torch.nn.ModuleList([torch.nn.Linear(params['dim_latent'], 1,bias = False) for i in range(params['bi_linear_layers'])])
                        self.BlinearCell2_out = torch.nn.ModuleList([torch.nn.Linear(params['dim_latent'], 1,bias = False) for i in range(params['bi_linear_layers'])])
                        augmented_size    = params['bi_linear_layers'] + params['dim_hidden_linear']
                        self.transLayers_out = torch.nn.ModuleList([torch.nn.Linear(augmented_size, params['dim_observations'])])
                        self.transLayers_out.extend([torch.nn.Linear(params['dim_observations'], params['dim_observations']) for i in range(1, params['transition_layers'])])

                    # def set_Markovian_NN(self):
                    #     self.linearCell_mark   = torch.nn.Linear(params['dim_observations'], params['dim_hidden_linear'])
                    #     self.BlinearCell1_mark = torch.nn.ModuleList([torch.nn.Linear(params['dim_observations'], 1,bias = False) for i in range(params['bi_linear_layers'])])
                    #     self.BlinearCell2_mark = torch.nn.ModuleList([torch.nn.Linear(params['dim_observations'], 1,bias = False) for i in range(params['bi_linear_layers'])])
                    #     augmented_size    = params['bi_linear_layers'] + params['dim_hidden_linear']
                    #     self.transLayers_mark = torch.nn.ModuleList([torch.nn.Linear(augmented_size, params['dim_observations'])])
                    #     self.transLayers_mark.extend([torch.nn.Linear(params['dim_observations'], params['dim_observations']) for i in range(1, params['transition_layers'])])

                    def f_nn_hidden(self, aug_inp):
                        # aug_inp is [x_k, r_k]
                        BP_outp = (torch.zeros((aug_inp.size()[0],params['bi_linear_layers'])))
                        L_outp   = self.linearCell(aug_inp)
                        for i in range((params['bi_linear_layers'])):
                            BP_outp[:,i]=self.BlinearCell1[i](aug_inp)[:,0]*self.BlinearCell2[i](aug_inp)[:,0]
                        aug_vect = torch.cat((L_outp, BP_outp), dim=1)
                        for i in range((params['transition_layers'])):
                            aug_vect = (self.transLayers[i](aug_vect))
                        return aug_vect#self.outputLayer(aug_vect)

                    def f_nn_output(self, r_k):
                        BP_outp = (torch.zeros((r_k.size()[0],params['bi_linear_layers'])))
                        L_outp   = self.linearCell_out(r_k)
                        for i in range((params['bi_linear_layers'])):
                            BP_outp[:,i]=self.BlinearCell1_out[i](r_k)[:,0]*self.BlinearCell2_out[i](r_k)[:,0]
                        aug_vect = torch.cat((L_outp, BP_outp), dim=1)
                        for i in range((params['transition_layers'])):
                            aug_vect = (self.transLayers_out[i](aug_vect))
                        return aug_vect#self.outputLayer(aug_vect)

                    # def f_nn_Mark(self, aug_inp):
                    #     BP_outp = (torch.zeros((aug_inp.size()[0],params['bi_linear_layers'])))
                    #     L_outp   = self.linearCell_mark(aug_inp)
                    #     for i in range((params['bi_linear_layers'])):
                    #         BP_outp[:,i]=self.BlinearCell1_mark[i](aug_inp)[:,0]*self.BlinearCell2_mark[i](aug_inp)[:,0]
                    #     aug_vect = torch.cat((L_outp, BP_outp), dim=1)
                    #     for i in range((params['transition_layers'])):
                    #         aug_vect = (self.transLayers_mark[i](aug_vect))
                    #     return aug_vect#self.outputLayer(aug_vect)

                    def rhs(self, aug_inp, dt):
                        '''This function is exclusively for solving the system post-training. For use with solve_ivp.'''
                        aug_inp = torch.FloatTensor(aug_inp).reshape(1,-1)
                        grad = torch.zeros(aug_inp.shape)
                        if self.use_f0:
                            f0 = self.get_mech(aug_inp)
                            grad += f0

                        if self.doMark:
                            mm = self.get_MarkClosure(aug_inp)
                            grad += mm

                        if self.doNonMark:
                            rr = self.get_NonMarkClosure(aug_inp)
                            grad += rr

                        return np.squeeze(grad.data.numpy())

                    def forward(self, inp, dt):
                        aug_inp = self.get_aug(inp)

                        grad = torch.zeros_like(aug_inp)
                        if self.use_f0:
                            f0 = self.get_mech(inp)
                            grad += f0

                        if self.doMark:
                            mm = self.get_MarkClosure(inp)
                            grad += mm

                        if self.doNonMark:
                            rr = self.get_NonMarkClosure(aug_inp)
                            grad += rr

                        return grad, aug_inp

                    def get_aug(self, inp):
                        if inp.shape[-1] < self.K * (self.D + 1):
                            aug_inp = torch.cat((inp, self.y_aug), dim=1)
                        else:
                            aug_inp = inp
                        return aug_inp

                    def get_mech(self, aug_inp):
                        X = aug_inp[:,:self.K]
                        f0 = torch.zeros(aug_inp.shape)
                        f0[:,:self.K] = self.f0(aug_inp[:,:self.K].T, 0).T
                        # print('||f0||^2 = ',torch.mean(f0**2))
                        return f0

                    def get_MarkClosure(self, inp):
                        g = torch.zeros_like(inp)
                        for k in range(self.K):
                            g[:,k,None] = self.f_nn_Mark(inp[:,k,None])
                        return g

                    def get_NonMarkClosure(self, aug_inp):
                        g = torch.zeros_like(aug_inp)
                        for k in range(self.K):
                            r_k = aug_inp[:,self.K + k*self.D: self.K + (k+1)*self.D]
                            g[:,k] = torch.squeeze(self.f_nn_output(r_k)) # store observed component

                            aug_inp_k = torch.cat((aug_inp[:,k,None], r_k), dim=1)
                            foo = self.f_nn_hidden(aug_inp_k)
                            g[:, self.K + k*self.D: self.K + (k+1)*self.D] = foo # store un-observed components
                        return g

                    def forward_true(self, inp, dt):
                        aug_inp = self.get_aug(inp)
                        grad = self.get_mech(inp)
                        return grad, aug_inp

                    def forward_approx(self, inp, dt):
                        """
                        In the forward function we accept a Tensor of input data and we must return
                        a Tensor of output data. We can use Modules defined in the constructor as
                        well as arbitrary operators on Tensors.
                        """
                        aug_inp = self.get_aug(inp)

                        grad = self.get_NonMarkClosure(aug_inp)
                        if self.use_f0:
                            f0 = self.get_mech(inp)
                            grad = grad + f0

                        return grad, aug_inp


        model  = FC_net(params)

        class INT_net(torch.nn.Module):
                def __init__(self, params):
                    super(INT_net, self).__init__()
        #            self.add_module('Dyn_net',FC_net(params))
                    # if self.f0only:
                    #     self.Dyn_net = model.forward_true
                    # else:
                    self.Dyn_net = model
                def forward(self, inp, dt):
                        k1, aug_inp   = self.Dyn_net(inp,dt)
                        inp_k2 = inp + 0.5*dt*k1
                        k2, tmp   = self.Dyn_net(inp_k2,dt)
                        inp_k3 = inp + 0.5*dt*k2
                        k3, tmp   = self.Dyn_net(inp_k3,dt)
                        inp_k4 = inp + dt*k3
                        k4, tmp   = self.Dyn_net(inp_k4,dt)
                        pred = aug_inp +dt*(k1+2*k2+2*k3+k4)/6
                        return pred, k1, inp, aug_inp
        modelRINN = INT_net(params)
        return model, modelRINN


def train_NbedDyn_model_L96MS(params,model,modelRINN,X_train,Grad_t,nm=''):
        dt = params['dt_integration']
        aug_inp_data = []
        x = torch.from_numpy(X_train).float()
        z = torch.from_numpy(Grad_t).float()
        d_inp = params['dim_input']

        criterion = torch.nn.MSELoss(reduction='none')

        # first, compute loss when using only f0 and no Neural Network
        # grad = modelRINN.Dyn_net.get_mech(x[b,:,:])
        if modelRINN.Dyn_net.f0only:
            for b in range(params['nb_batch']):
                # Forward pass: Compute predicted y by passing x to the model
                inp_concat = torch.cat((x[b,:,:], modelRINN.Dyn_net.y_aug[b,:,:]), dim=1)
                print('Forward 1')
                pred, grad, inp, aug_inp     = modelRINN(inp_concat,dt)
                print('Forward 2')
                pred2, grad2, inp2, aug_inp2 = modelRINN(pred,dt)
                # Compute and print loss
                loss1 = criterion(grad[:,:d_inp], z[b,:,:]).mean()
                loss2 = criterion(pred[:-1,d_inp:] , aug_inp[1:,d_inp:]).sum()
                loss3 = criterion(pred2[:-1,d_inp:] , pred[1:,d_inp:]).sum()
                loss =  0.1*loss1 + 0.9*loss2 + 0.9*loss3
                print('||grad||^2 = ', torch.mean(grad**2))
                print('Only f0 Loss ', loss)
                print('Only f0 rhs loss ', loss1)
            return model, modelRINN, aug_inp_data

        if params['pretrained'] :
            modelRINN.load_state_dict(torch.load(params['path'] + params['file_name']+'.pt'))
            lr = 0.01
        else:
            lr = 0.1
            # print('Training model...')
            try:
                modelRINN.load_state_dict(torch.load(params['path'] + params['file_name']+'.pt'))
                print('Loaded trained model!')
                for b in range(params['nb_batch']):
                    # Forward pass: Compute predicted y by passing x to the model
                    inp_concat = torch.cat((x[b,:,:], modelRINN.Dyn_net.y_aug[b,:,:]), dim=1)
                    print('Forward 1')
                    pred, grad, inp, aug_inp     = modelRINN(inp_concat,dt)
                    print('Forward 2')
                    pred2, grad2, inp2, aug_inp2 = modelRINN(pred,dt)
                    # Compute and print loss
                    loss1 = criterion(grad[:,:d_inp], z[b,:,:]).mean()
                    loss2 = criterion(pred[:-1,d_inp:] , aug_inp[1:,d_inp:]).sum()
                    loss3 = criterion(pred2[:-1,d_inp:] , pred[1:,d_inp:]).sum()
                    loss =  0.1*loss1 + 0.9*loss2 + 0.9*loss3
                    print('||grad||^2 = ', torch.mean(grad**2))
                    print('Trained model Loss ', loss)
                    print('Trained model rhs loss ', loss1)
                return model, modelRINN, aug_inp_data
            except:
                print('Training model...')
                pass

        optimizer = torch.optim.Adam(model.parameters())

        for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        loss_vec1 = {'train': [], 'validate': []}
        N = x.shape[1]
        N_train = N #int(0.9*N)
        best_loss = np.inf
        for b in range(params['nb_batch']):
            for t in range(0,params['ntrain'][0]):
                # Forward pass: Compute predicted y by passing x to the model
                inp_concat = torch.cat((x[b,:,:], modelRINN.Dyn_net.y_aug[b,:,:]), dim=1)
                pred, grad, inp, aug_inp     = modelRINN(inp_concat,dt)
                pred2, grad2, inp2, aug_inp2 = modelRINN(pred,dt)
                if params['get_latent_train']:
                    aug_inp_data.append(aug_inp.detach())
                # Compute and print loss
                # bp()
                foo1 = criterion(grad[:,:d_inp], z[b,:,:])
                loss1_train = foo1[:N_train].mean()
                loss1_validate = foo1[N_train:].mean()

                foo2 = criterion(pred[:-1,d_inp:] , aug_inp[1:,d_inp:])
                loss2_train = foo2[:N_train].sum()
                loss2_validate = foo2[N_train:].sum()

                foo3 = criterion(pred2[:-1,d_inp:] , pred[1:,d_inp:])
                loss3_train = foo3[:N_train].sum()
                loss3_validate = foo3[N_train:].sum()

                loss =  0.1*loss1_train + 0.9*loss2_train + 0.9*loss3_train
                # loss_validate = 0.1*loss1_validate + 0.9*loss2_validate + 0.9*loss3_validate

                loss_vec1['train'].append(loss.data.numpy())
                # loss_vec1['validate'].append(loss_validate.data.numpy())
                if t%10==0:
                    print('Training Loss:', loss, 'at epoch', t)
                    # print('Validation Loss:', loss_validate, 'at epoch', t)
                    plot_training_progress(loss_vec1, os.path.join(params['path'],'training_part1_{}'.format(nm)))
                    # bp()

                if loss.detach().data.numpy() < best_loss:
                    best_loss = loss.data.numpy()
                    torch.save(modelRINN.state_dict(), params['path'] + params['file_name']+'_best.pt')

                torch.save(modelRINN.state_dict(), params['path'] + params['file_name']+'.pt')
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                if t==250:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 0.01
                if t==3000:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 0.001

        # plot final training losses, part 1
        plot_training_progress(loss_vec1, os.path.join(params['path'],'training_part1_{}'.format(nm)))

        print('Part 2 of training...')
        loss_vec2 = {'train': [], 'validate': []}
        for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
        for t in range(0,params['ntrain'][1]):
            for b in range(params['nb_batch']):
                # Forward pass: Compute predicted y by passing x to the model
                inp_concat = torch.cat((x[b,:,:], modelRINN.Dyn_net.y_aug[b,:,:]), dim=1)
                pred, grad, inp, aug_inp = modelRINN(inp_concat,dt)
                # Compute and print loss

                foo1 = criterion(grad[:,:d_inp], z[b,:,:])
                loss1_train = foo1[:N_train].mean()
                loss1_validate = foo1[N_train:].mean()

                foo2 = criterion(pred[:-1,:d_inp] , aug_inp[1:,:d_inp])
                loss2_train = foo2[:N_train].sum()
                loss2_validate = foo2[N_train:].sum()

                foo3 = criterion(pred[:-1,d_inp:] , pred[1:,d_inp:])
                loss3_train = foo3[:N_train].sum()
                loss3_validate = foo3[N_train:].sum()

                loss =  0.0*loss1_train + 1.0*loss2_train + 1.0*loss3_train
                loss_validate = 0.0*loss1_validate + 1.0*loss2_validate + 1.0*loss3_validate
                loss_vec2['train'].append(loss.data.numpy())
                # loss_vec2['validate'].append(loss_validate.data.numpy())
                if t%10==0:
                    print('Training Loss:', loss, 'at epoch', t)
                    plot_training_progress(loss_vec2, os.path.join(params['path'],'training_part2_{}'.format(nm)))
                    # print('Validation Loss:', loss_validate, 'at epoch', t)
                torch.save(modelRINN.state_dict(), params['path'] + params['file_name']+'.pt')
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        # plot final training losses, part 2
        plot_training_progress(loss_vec2, os.path.join(params['path'],'training_part2_{}'.format(nm)))
        torch.save(modelRINN.state_dict(), params['path'] + params['file_name'])
        return model, modelRINN, aug_inp_data


def train_NbedDyn_model_L63(params,model,modelRINN,X_train,Grad_t):
        dt = params['dt_integration']
        aug_inp_data = []
        x = torch.from_numpy(X_train).float()
        z = torch.from_numpy(Grad_t).float()
        d_inp = params['dim_input']

        criterion = torch.nn.MSELoss(reduction='none')

        if params['pretrained'] :
            modelRINN.load_state_dict(torch.load(params['path'] + params['file_name']+'.pt'))


        try:
            modelRINN.load_state_dict(torch.load(params['path'] + params['file_name']+'.pt'))
            print('Loaded trained model!')
            return model, modelRINN, aug_inp_data
        except:
            print('Training model...')
            pass

        optimizer = torch.optim.Adam(model.parameters())

        for param_group in optimizer.param_groups:
                print('lr =', param_group['lr'])
        for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1

        for t in range(0,params['ntrain'][0]):
            for b in range(params['nb_batch']):
                # Forward pass: Compute predicted y by passing x to the model
                inp_concat = torch.cat((x[b,:,:], modelRINN.Dyn_net.y_aug[b,:,:]), dim=1)
                pred, grad, inp, aug_inp     = modelRINN(inp_concat,dt)
                pred2, grad2, inp2, aug_inp2 = modelRINN(pred,dt)
                if params['get_latent_train']:
                    aug_inp_data.append(aug_inp.detach())
                # Compute and print loss
                # bp()
                loss1 = criterion(grad[:,:d_inp], z[b,:,:]).mean()
                loss2 = criterion(pred[:-1,d_inp:] , aug_inp[1:,d_inp:]).sum()
                loss3 = criterion(pred2[:-1,d_inp:] , pred[1:,d_inp:]).sum()
                loss =  0.1*loss1+0.9*loss2 + 0.9*loss3
                if t%1000==0:
                    print('Training L63 NbedDyn model', t,loss)
                torch.save(modelRINN.state_dict(), params['path'] + params['file_name']+'.pt')
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            if t>1500:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.01
            if t>5500:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.001

        for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
        for t in range(0,params['ntrain'][1]):
            for b in range(params['nb_batch']):
                # Forward pass: Compute predicted y by passing x to the model
                inp_concat = torch.cat((x[b,:,:], modelRINN.Dyn_net.y_aug[b,:,:]), dim=1)
                pred, grad, inp, aug_inp = modelRINN(inp_concat,dt)
                # Compute and print loss
                loss1 = criterion(grad[:,:d_inp], z[b,:,:]).mean()
                loss2 = criterion(pred[:-1,:d_inp] , aug_inp[1:,:d_inp]).sum()
                loss3 = criterion(pred[:-1,d_inp:] , aug_inp[1:,d_inp:]).sum()
                loss =  0.0*loss1+1.0*loss2 + 1.0*loss3
                if t%1000==0:
                    print('Training L63 NbedDyn model', t,loss)
                torch.save(modelRINN.state_dict(), params['path'] + params['file_name']+'.pt')
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        torch.save(modelRINN.state_dict(), params['path'] + params['file_name'])
        return model, modelRINN, aug_inp_data
def train_NbedDyn_model_SLA(params,model,modelRINN,X_train,Grad_t):
        dt = params['dt_integration']
        aug_inp_data = []
        x = torch.from_numpy(X_train).float()
        z = torch.from_numpy(Grad_t).float()


        criterion = torch.nn.MSELoss(reduction='none')

        if params['pretrained'] :
            modelRINN.load_state_dict(torch.load(params['path'] + params['file_name']+'.pt'))
        optimizer = torch.optim.Adam(model.parameters())

        for param_group in optimizer.param_groups:
                print('lr = ', param_group['lr'])
        for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001

        for t in range(params['ntrain'][0]):
            for b in range(params['nb_batch']):
                optimizer.zero_grad()
                inp_concat = torch.cat((x[b,:,:], modelRINN.Dyn_net.y_aug[b,:,:]), dim=1)
                pred1, grad1, inp, aug_inp = modelRINN(inp_concat,dt)
                if params['get_latent_train']:
                    aug_inp_data.append(aug_inp.detach())
                loss1 = criterion(grad1[:,:params['dim_input']], z[b,:,:]).mean()
                loss2 = criterion(pred1[:-1,:], inp_concat[1:,:]).sum()
                loss = 0.9*loss1+0.1*loss2
                if t%1000==0:
                        print('Training SLA NbedDyn model', t,loss)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        torch.save(modelRINN.state_dict(), params['path'] + params['file_name']+'.pt')
        return model, modelRINN, aug_inp_data
def train_NbedDyn_model_Linear(params,model,modelRINN,X_train,Grad_t):
        dt = params['dt_integration']
        aug_inp_data = []
        x = torch.from_numpy(X_train).float()
        z = torch.from_numpy(Grad_t).float()


        criterion = torch.nn.MSELoss(reduction='none')

        if params['pretrained'] :
            modelRINN.load_state_dict(torch.load(params['path'] + params['file_name']+'.pt'))
        optimizer = torch.optim.Adam(model.parameters())

        for param_group in optimizer.param_groups:
                print('lr = ', param_group['lr'])
        for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1

        for t in range(params['ntrain'][0]):
            # Forward pass: Compute predicted y by passing x to the model
            for b in range(params['nb_batch']):
                aug_inp = torch.cat((x[b,:,:],modelRINN.Dyn_net.y_aug[b,:,:]),dim = -1)
                pred, grad, inp, aug_inp = modelRINN(aug_inp,dt)
                if params['get_latent_train']:
                    aug_inp_data.append(aug_inp.detach())
                # Compute and print loss
                loss1 = criterion(grad[:,:1], z[b,:,:]).mean()
                loss2 = criterion(pred[:-1,:], aug_inp[1:,:]).sum()
                loss = 1.0*loss1+1.0*loss2
                if t%1000==0:
                       print('Training Linear NbedDyn model', t,loss)
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        torch.save(modelRINN.state_dict(), params['path'] + params['file_name']+'.pt')
        return model, modelRINN, aug_inp_data
