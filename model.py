import numpy as np 
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model_utils import *


def process_input_data(input_type, data, macro, params):
	if input_type == 'y':
		return data
	elif input_type == 'xy':
		# Get the trained offense and defense for x
		n_agents = params['n_agents']
		n_trained_off = max(params['n_trained_off'], n_agents)
		n_gt_off = params['n_gt_off']
		n_trained_def = params['n_trained_def']
		n_offense = n_trained_off + n_gt_off
		trained_indices = [i for i in range(2 * n_trained_off)] + \
			[i for i in range(2 * n_offense, 2 * (n_offense + n_trained_def))]
		n_trained_players = n_trained_off + n_trained_def

		x = data[1:,:, trained_indices].clone()
		x = x.view(x.size(0), x.size(1), n_trained_players, -1).transpose(1,2)
		y = data
		return x, y
	elif input_type == 'xym':
		# Get the trained offense and defense for x
		n_agents = params['n_agents']
		n_trained_off = max(params['n_trained_off'], n_agents)
		n_gt_off = params['n_gt_off']
		n_trained_def = params['n_trained_def']
		n_offense = n_trained_off + n_gt_off
		trained_indices = [i for i in range(2 * n_trained_off)] + \
			[i for i in range(2 * n_offense, 2 * (n_offense + n_trained_def))]
		n_trained_players = n_trained_off + n_trained_def

		x = data[1:,:, trained_indices].clone()
		x = x.view(x.size(0), x.size(1), n_trained_players, -1).transpose(1,2)
		y = data

		macro_ohe = torch.zeros(data.size(0), n_trained_players, data.size(1), 90)
		for i in range(n_trained_players):
			macro_ohe[:,i,:,:] = one_hot_encode(macro[:,:,i].data, 90)
		macro_ohe = Variable(macro_ohe).cuda() if macro.is_cuda else Variable(macro_ohe)

		return x, y, macro_ohe
	else:
		return data, macro


def num_trainable_params(model):
	total = 0
	for p in model.parameters():
		count = 1
		for s in p.size():
			count *= s
		total += count
	return total


def cudafy_list(states):
	for i in range(len(states)):
		states[i] = states[i].cuda()
	return states


# 104
class MACRO_VRNN(nn.Module):

	def __init__(self, params):
		super(MACRO_VRNN, self).__init__()

		self.input_type = 'xym'
		self.params = params
		x_dim = params['x_dim']
		y_dim = params['y_dim']
		z_dim = params['z_dim']
		h_dim = params['h_dim']
		m_dim = params['m_dim']
		rnn_micro_dim = params['rnn_micro_dim']
		rnn_macro_dim = params['rnn_macro_dim']
		n_layers = params['n_layers']
		n_agents = params['n_agents']
		n_trained_off = max(params['n_trained_off'], n_agents)
		n_trained_def = params['n_trained_def']
		n_trained_players = n_trained_def + n_trained_off

		# takes the y, rnn_macro, and outputs m
		self.dec_macro = nn.ModuleList([nn.Sequential(
			nn.Linear(y_dim+rnn_macro_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, m_dim),
			nn.LogSoftmax()) for i in range(n_trained_players)])

		# takes the x, m, rnn_micro, and outputs h
		self.enc = nn.ModuleList([nn.Sequential(
			nn.Linear(x_dim+m_dim+rnn_micro_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU()) for i in range(n_trained_players)])

		# Takes h, outputs z
		self.enc_mean = nn.ModuleList([nn.Linear(h_dim, z_dim) for i in range(n_trained_players)])

		# Takes h, outputs z
		self.enc_std = nn.ModuleList([nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus()) for i in range(n_trained_players)])

		# Takes m_dim, and rnn_micro (no x_dim), outputs h
		# Otherwise, dims are same as enc
		self.prior = nn.ModuleList([nn.Sequential(
			nn.Linear(m_dim+rnn_micro_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU()) for i in range(n_trained_players)])
		self.prior_mean = nn.ModuleList([nn.Linear(h_dim, z_dim) for i in range(n_trained_players)])
		self.prior_std = nn.ModuleList([nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus()) for i in range(n_trained_players)])

		# takes y, m, z, rnn_micro, outputs h_dim
		self.dec = nn.ModuleList([nn.Sequential(
			nn.Linear(y_dim+m_dim+z_dim+rnn_micro_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU()) for i in range(n_trained_players)])

		# takes h, outputs x
		self.dec_mean = nn.ModuleList([nn.Linear(h_dim, x_dim) for i in range(n_trained_players)])
		self.dec_std = nn.ModuleList([nn.Sequential(
			nn.Linear(h_dim, x_dim),
			nn.Softplus()) for i in range(n_trained_players)])

		# takes x and z, outputs rnn_micro
		self.gru_micro = nn.ModuleList([nn.GRU(x_dim+z_dim, rnn_micro_dim, n_layers)
			for i in range(n_trained_players)])
		# takes m * n, outputs rnn_macro
		self.gru_macro = nn.GRU(m_dim*n_trained_players, rnn_macro_dim, n_layers)


	def forward(self, data, macro=None, hp=None):
		x, y, m = process_input_data(self.input_type, data, macro, self.params)

		out = {}
		out['recon_loss'] = 0

		if not hp['pretrain']:
			out['kl_loss'] = 0

		n_agents = self.params['n_agents']
		n_trained_off = max(self.params['n_trained_off'], n_agents)
		n_trained_def = self.params['n_trained_def']
		n_trained_players = n_trained_def + n_trained_off
		
		h_micro = [Variable(torch.zeros(self.params['n_layers'], y.size(1), self.params['rnn_micro_dim']))
			for i in range(n_trained_players)]
		h_macro = Variable(torch.zeros(self.params['n_layers'], y.size(1), self.params['rnn_macro_dim']))
		if self.params['cuda']:
			h_macro = h_macro.cuda()
			h_micro = cudafy_list(h_micro)

		for t in range(y.size(0)-1):
			x_t = x[t].clone()
			y_t = y[t].clone()
			m_t = m[t].clone()
			
			if hp['pretrain']:
				for i in range(n_trained_players):
					dec_macro_t = self.dec_macro[i](torch.cat([y_t, h_macro[-1]], 1))
					out['recon_loss'] -= torch.sum(m_t[i]*dec_macro_t)
	
				m_t_concat = m_t.transpose(0,1).contiguous().view(y.size(1), -1).clone()
				_, h_macro = self.gru_macro(torch.cat([m_t_concat], 1).unsqueeze(0), h_macro)

			else:
				for i in range(n_trained_players):
					enc_t = self.enc[i](torch.cat([x_t[i], m_t[i], h_micro[i][-1]], 1))
					enc_mean_t = self.enc_mean[i](enc_t)
					enc_std_t = self.enc_std[i](enc_t)

					prior_t = self.prior[i](torch.cat([m_t[i], h_micro[i][-1]], 1))
					prior_mean_t = self.prior_mean[i](prior_t)
					prior_std_t = self.prior_std[i](prior_t)

					z_t = sample_gauss(enc_mean_t, enc_std_t)

					dec_t = self.dec[i](torch.cat([y_t, m_t[i], z_t, h_micro[i][-1]], 1))
					dec_mean_t = self.dec_mean[i](dec_t)
					dec_std_t = self.dec_std[i](dec_t)

					_, h_micro[i] = self.gru_micro[i](torch.cat([x_t[i], z_t], 1).unsqueeze(0), h_micro[i])

					out['kl_loss'] += kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
					out['recon_loss'] += nll_gauss(dec_mean_t, dec_std_t, x_t[i])

		return out


	def sample(self, data, macro, seq_len=0, burn_in=0, fix_m=[], seqs_per_sample=1):
		x, y, m = process_input_data(self.input_type, data, macro, self.params)

		n_agents = self.params['n_agents']
		n_trained_off = max(self.params['n_trained_off'], n_agents)
		n_trained_def = self.params['n_trained_def']
		n_trained_players = n_trained_def + n_trained_off

		if seq_len == 0:
			seq_len = y.size(0)-1

		if len(fix_m) == 0:
			fix_m = [-1]*n_trained_players

		h_micro = [Variable(torch.zeros(self.params['n_layers'], y.size(1), self.params['rnn_micro_dim']))
			for i in range(n_trained_players)]
		h_macro = Variable(torch.zeros(self.params['n_layers'], y.size(1), self.params['rnn_macro_dim']))

		macro_goals = Variable(torch.zeros(seq_len+1, y.size(1), n_trained_players))
		if self.params['cuda']:
			h_macro, macro_goals = h_macro.cuda(), macro_goals.cuda()
			h_micro = cudafy_list(h_micro)

		all_rets = []
		all_macro_goals = []

		for _ in range(seqs_per_sample):
			ret = y.clone()
			for t in range(seq_len):
				y_t = ret[t].clone()
				m_t = m[t].clone()

				# Since i is looping over the first n agents, when train_def is
				# turned on, and n_agents = 5,
				# the defense is not affected by the sampling procedure.

				for i in range(n_trained_players):
					dec_macro_t = self.dec_macro[i](torch.cat([y_t, h_macro[-1]], 1))
					m_t[i] = sample_multinomial(torch.exp(dec_macro_t))

				macro_goals[t] = torch.max(m_t, 2)[1].transpose(0,1)
				m_t_concat = m_t.transpose(0,1).contiguous().view(y.size(1), -1)
				_, h_macro = self.gru_macro(torch.cat([m_t_concat], 1).unsqueeze(0), h_macro)

				for i in range(n_trained_players):
					prior_t = self.prior[i](torch.cat([m_t[i], h_micro[i][-1]], 1))
					prior_mean_t = self.prior_mean[i](prior_t)
					prior_std_t = self.prior_std[i](prior_t)

					z_t = sample_gauss(prior_mean_t, prior_std_t)

					dec_t = self.dec[i](torch.cat([y_t, m_t[i], z_t, h_micro[i][-1]], 1))
					dec_mean_t = self.dec_mean[i](dec_t)
					dec_std_t = self.dec_std[i](dec_t)

					# ret[t+1,:,2*i:2*i+2] = y[t+1,:,2*i:2*i+2] if t < burn_in or i > 0 else sample_gauss(dec_mean_t, dec_std_t)
					ret[t+1,:,2*i:2*i+2] = y[t+1,:,2*i:2*i+2] if t < burn_in else sample_gauss(dec_mean_t, dec_std_t)
					_, h_micro[i] = self.gru_micro[i](torch.cat([ret[t+1,:,2*i:2*i+2], z_t], 1).unsqueeze(0), h_micro[i])
			print(macro_goals.shape, macro_goals.data.cpu().numpy().shape)
			macro_goals.data[-1] = macro_goals.data[-2]
			all_rets.append(ret.data.cpu().numpy())
			all_macro_goals.append(macro_goals.data.cpu().numpy())
		print(np.array(all_rets).shape, np.array(all_macro_goals).shape)
		return np.array(all_rets), np.array(all_macro_goals)

# 103
class VRNN_INDEP(nn.Module):

	def __init__(self, params):
		super(VRNN_INDEP, self).__init__()

		self.input_type = 'y'
		self.params = params
		x_dim = params['x_dim']
		y_dim = params['y_dim']
		z_dim = params['z_dim']
		h_dim = params['h_dim']
		rnn_dim = params['rnn_dim']
		n_layers = params['n_layers']
		n_agents = params['n_agents']

		self.enc = nn.ModuleList([nn.Sequential(
			nn.Linear(x_dim+y_dim+rnn_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU()) for i in range(n_agents)])
		self.enc_mean = nn.ModuleList([nn.Linear(h_dim, z_dim) for i in range(n_agents)])
		self.enc_std = nn.ModuleList([nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus()) for i in range(n_agents)])

		self.prior = nn.ModuleList([nn.Sequential(
			nn.Linear(y_dim+rnn_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU()) for i in range(n_agents)])
		self.prior_mean = nn.ModuleList([nn.Linear(h_dim, z_dim) for i in range(n_agents)])
		self.prior_std = nn.ModuleList([nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus()) for i in range(n_agents)])

		self.dec = nn.ModuleList([nn.Sequential(
			nn.Linear(y_dim+z_dim+rnn_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU()) for i in range(n_agents)])
		self.dec_mean = nn.ModuleList([nn.Linear(h_dim, x_dim) for i in range(n_agents)])
		self.dec_std = nn.ModuleList([nn.Sequential(
			nn.Linear(h_dim, x_dim),
			nn.Softplus()) for i in range(n_agents)])

		self.rnn = nn.ModuleList([nn.GRU(x_dim+z_dim, rnn_dim, n_layers) for i in range(n_agents)])


	def forward(self, data, macro=None, hp=None):
		y = process_input_data(self.input_type, data, macro, self.params)

		out = {}
		out['kl_loss'] = 0
		out['recon_loss'] = 0

		n_agents = int(self.params['y_dim']/self.params['x_dim'])

		h = [Variable(torch.zeros(self.params['n_layers'], y.size(1), self.params['rnn_dim'])) for i in range(n_agents)]
		if self.params['cuda']:
			h = cudafy_list(h)

		for t in range(y.size(0)-1):
			y_t = y[t].clone()

			for i in range(n_agents):
				x_t = y[t+1][:,2*i:2*i+2].clone()

				enc_t = self.enc[i](torch.cat([x_t, y_t, h[i][-1]], 1))
				enc_mean_t = self.enc_mean[i](enc_t)
				enc_std_t = self.enc_std[i](enc_t)

				prior_t = self.prior[i](torch.cat([y_t, h[i][-1]], 1))
				prior_mean_t = self.prior_mean[i](prior_t)
				prior_std_t = self.prior_std[i](prior_t)

				z_t = sample_gauss(enc_mean_t, enc_std_t)

				dec_t = self.dec[i](torch.cat([y_t, z_t, h[i][-1]], 1))
				dec_mean_t = self.dec_mean[i](dec_t)
				dec_std_t = self.dec_std[i](dec_t)

				_, h[i] = self.rnn[i](torch.cat([x_t, z_t], 1).unsqueeze(0), h[i])

				out['kl_loss'] += kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
				out['recon_loss'] += nll_gauss(dec_mean_t, dec_std_t, x_t)

		return out
	

	def sample(self, data, macro, seq_len=0, burn_in=0):
		y = process_input_data(self.input_type, data, macro, self.params)

		if seq_len == 0:
			seq_len = y.size(0)-1

		n_agents = int(self.params['y_dim']/self.params['x_dim'])

		h = [Variable(torch.zeros(self.params['n_layers'], y.size(1), self.params['rnn_dim'])) for i in range(n_agents)]
		if self.params['cuda']:
			h = cudafy_list(h)

		ret = y.clone()

		for t in range(seq_len):
			y_t = ret[t].clone()

			for i in range(n_agents):
				prior_t = self.prior[i](torch.cat([y_t, h[i][-1]], 1))
				prior_mean_t = self.prior_mean[i](prior_t)
				prior_std_t = self.prior_std[i](prior_t)

				z_t = sample_gauss(prior_mean_t, prior_std_t)

				dec_t = self.dec[i](torch.cat([y_t, z_t, h[i][-1]], 1))
				dec_mean_t = self.dec_mean[i](dec_t)
				dec_std_t = self.dec_std[i](dec_t)

				if t >= burn_in:
					ret[t+1,:,2*i:2*i+2] = sample_gauss(dec_mean_t, dec_std_t)

				_, h[i] = self.rnn[i](torch.cat([ret[t+1,:,2*i:2*i+2], z_t], 1).unsqueeze(0), h[i])		

		return ret

# 102
class VRNN_SINGLE(nn.Module):

	def __init__(self, params):
		super(VRNN_SINGLE, self).__init__()

		self.input_type = 'y'
		self.params = params
		y_dim = params['y_dim']
		z_dim = params['z_dim']
		h_dim = params['h_dim']
		rnn_dim = params['rnn_dim']
		n_layers = params['n_layers']

		self.enc = nn.Sequential(
			nn.Linear(y_dim+y_dim+rnn_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.enc_mean = nn.Linear(h_dim, z_dim)
		self.enc_std = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus())

		self.prior = nn.Sequential(
			nn.Linear(y_dim+rnn_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.prior_mean = nn.Linear(h_dim, z_dim)
		self.prior_std = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus())

		self.dec = nn.Sequential(
			nn.Linear(y_dim+z_dim+rnn_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.dec_mean = nn.Linear(h_dim, y_dim)
		self.dec_std = nn.Sequential(
			nn.Linear(h_dim, y_dim),
			nn.Softplus())

		self.rnn = nn.GRU(y_dim+z_dim, rnn_dim, n_layers)


	def forward(self, data, macro=None, hp=None):
		y = process_input_data(self.input_type, data, macro, self.params)

		out = {}
		out['kl_loss'] = 0
		out['recon_loss'] = 0

		h = Variable(torch.zeros(self.params['n_layers'], y.size(1), self.params['rnn_dim']))
		if self.params['cuda']:
			h = h.cuda()			
		
		for t in range(y.size(0)-1):

			y_t = y[t].clone()
			x_t = y[t+1].clone()

			enc_t = self.enc(torch.cat([x_t, y_t, h[-1]], 1))
			enc_mean_t = self.enc_mean(enc_t)
			enc_std_t = self.enc_std(enc_t)

			prior_t = self.prior(torch.cat([y_t, h[-1]], 1))
			prior_mean_t = self.prior_mean(prior_t)
			prior_std_t = self.prior_std(prior_t)

			z_t = sample_gauss(enc_mean_t, enc_std_t)

			dec_t = self.dec(torch.cat([y_t, z_t, h[-1]], 1))
			dec_mean_t = self.dec_mean(dec_t)
			dec_std_t = self.dec_std(dec_t)

			_, h = self.rnn(torch.cat([x_t, z_t], 1).unsqueeze(0), h)

			out['kl_loss'] += kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
			out['recon_loss'] += nll_gauss(dec_mean_t, dec_std_t, x_t)

		return out
	

	def sample(self, data, macro, seq_len=0, burn_in=0):
		y = process_input_data(self.input_type, data, macro, self.params)

		if seq_len == 0:
			seq_len = y.size(0)-1

		h = Variable(torch.zeros(self.params['n_layers'], y.size(1), self.params['rnn_dim']))
		if self.params['cuda']:
			h = h.cuda()

		for t in range(seq_len):
			y_t = y[t].clone()
			
			prior_t = self.prior(torch.cat([y_t, h[-1]], 1))
			prior_mean_t = self.prior_mean(prior_t)
			prior_std_t = self.prior_std(prior_t)

			z_t = sample_gauss(prior_mean_t, prior_std_t)

			dec_t = self.dec(torch.cat([y_t, z_t, h[-1]], 1))
			dec_mean_t = self.dec_mean(dec_t)
			dec_std_t = self.dec_std(dec_t)

			if t >= burn_in:
				y[t+1] = sample_gauss(dec_mean_t, dec_std_t)

			_, h = self.rnn(torch.cat([y[t+1], z_t], 1).unsqueeze(0), h)

		return y

# 101
class RNN_GAUSS(nn.Module):

	def __init__(self, params):
		super(RNN_GAUSS, self).__init__()

		self.input_type = 'y'
		self.params = params
		x_dim = params['x_dim']
		y_dim = params['y_dim']
		h_dim = params['h_dim']
		rnn_dim = params['rnn_dim']
		n_layers = params['n_layers']

		self.dec = nn.Sequential(
			nn.Linear(rnn_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.dec_mean = nn.Linear(h_dim, y_dim)
		self.dec_std = nn.Sequential(
			nn.Linear(h_dim, y_dim),
			nn.Softplus())

		self.rnn = nn.GRU(y_dim, rnn_dim, n_layers)


	def forward(self, data, macro=None, hp=None):
		y = process_input_data(self.input_type, data, macro, self.params)

		out = {}
		out['nll'] = 0

		h = Variable(torch.zeros(self.params['n_layers'], y.size(1), self.params['rnn_dim']))
		if self.params['cuda']:
			h = h.cuda()			
		
		for t in range(y.size(0)):
			y_t = y[t].clone()

			dec_t = self.dec(torch.cat([h[-1]], 1))
			dec_mean_t = self.dec_mean(dec_t)
			dec_std_t = self.dec_std(dec_t)

			_, h = self.rnn(y_t.unsqueeze(0), h)

			out['nll'] += nll_gauss(dec_mean_t, dec_std_t, y_t)

		return out
	
	
	def sample(self, data, macro, seq_len=0, burn_in=0):
		y = process_input_data(self.input_type, data, macro, self.params)

		if seq_len == 0:
			seq_len = y.size(0)

		h = Variable(torch.zeros(self.params['n_layers'], y.size(1), self.params['rnn_dim']))
		if self.params['cuda']:
			h = h.cuda()

		for t in range(seq_len):
			dec_t = self.dec(torch.cat([h[-1]], 1))
			dec_mean_t = self.dec_mean(dec_t)
			dec_std_t = self.dec_std(dec_t)

			if t >= burn_in:
				y[t] = sample_gauss(dec_mean_t, dec_std_t)

			_, h = self.rnn(y[t].unsqueeze(0), h)

		return y