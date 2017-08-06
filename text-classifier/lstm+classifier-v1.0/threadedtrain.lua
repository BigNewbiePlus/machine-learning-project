local Threads = require 'threads'
require 'base'
require 'nn'
require 'nngraph'

-- lstm setup for model and checkpoint
local function lstm_clone(pre_lstm_model, lstm_params)
   print("Creating a RNN LSTM network.")
   local lstm_model = {}
   lstm_model.s = {}
   lstm_model.ds = {}
   lstm_model.start_s = {}
   for j = 0, lstm_params.seq_length do
      lstm_model.s[j] = {}
      for d = 1, 2 * lstm_params.layers do
	 lstm_model.s[j][d] = torch.zeros(lstm_params.batch_size, lstm_params.rnn_size)
      end
   end
   for d = 1, 2 * lstm_params.layers do
      lstm_model.start_s[d] = torch.zeros(lstm_params.batch_size, lstm_params.rnn_size)
      lstm_model.ds[d] = torch.zeros(lstm_params.batch_size, lstm_params.rnn_size)
   end
   
   lstm_model.core_network = pre_lstm_model.core_network:clone('weight', 'bias')
   lstm_model.rnns = g_cloneManyTimes(lstm_model.core_network, lstm_params.seq_length)
   lstm_model.norm_dw = 0
   lstm_model.err = torch.zeros(lstm_params.seq_length)
   lstm_model.model_type = 'lstm'
   return lstm_model
end

local function mlp_clone(pre_mlp_model, mlp_params)
   local mlp_model = {}
   print("Creating a Multi-Layer Perception network.")
   local core_network = pre_mlp_model.core_network:clone('weight', 'bias')
   mlp_model.core_network = core_network
   mlp_model.norm_dw = 0
   mlp_model.err = 0
   mlp_model.model_type = 'mlp'
   return mlp_model
end

local function cp(pos, lr, step, epoch, total_cases, perps, tics, weights, cp_path)
   local cp_params = {}
   cp_params.pos = pos
   cp_params.lr = lr
   cp_params.step = step
   cp_params.epoch = epoch
   cp_params.total_cases = total_cases
   cp_params.perps = perps
   cp_params.tics = tics
   cp_params.weights = weights
   torch.save(cp_path, cp_params)
end

-- model clone, share parameters
local function model_clone(model, params)
   if model.model_type == 'lstm' then
      return lstm_clone(model, params)
   elseif model.model_type == 'mlp' then
      return mlp_clone(model, params)
   else
      print('no model, error!')
   end
end

local function reset_ds(ds)
   for d = 1, #ds do
      ds[d]:zero()
   end
end

local function fill(ds, value)
   for d = 1, #ds do
      ds[d]:fill(value)
   end
end


local function fp(dataset, idx, model, params)
   if model.model_type == 'lstm' then
      reset_ds(model.start_s)
      g_replace_table(model.s[0], model.start_s)

      local ex = dataset[idx]
      local x, y = ex[1], ex[2]
      for i = 1, params.seq_length do
	 local s = model.s[i - 1]
	 model.err[i], model.s[i] = table.unpack(model.rnns[i]:forward({x[i], y[i], s}))
      end  
      return model.err:mean()
   else
      local ex = dataset[idx]
      local x, y = ex[1], ex[2]
      local loss = model.core_network:forward({x, y})
      return loss[1]
   end
end

local function bp(dataset, idx, model, params)
   model.core_network:zeroGradParameters()
   if model.model_type == 'lstm' then
      --reset_ds(model.ds)
      fill(model.ds, 0)
      local ex = dataset[idx]
      local x, y = ex[1], ex[2]
      for i = params.seq_length, 1, -1 do
	 local derr = torch.ones(1)
	 local s = model.s[i-1]
	 local tmp = model.rnns[i]:backward({x[i], y[i], s},
	    {derr, model.ds})[3]
	 g_replace_table(model.ds, tmp)
      end
   else
      local ex = dataset[idx]
      local x,y = ex[1], ex[2]
      local derr = torch.ones(1)
      model.core_network:backward({x, y}, derr)
   end
end

local function build_dataset(model, data, label, params)
   local dataset = {}
   -- no label, lstm train
   if model.model_type == 'lstm' then
      function dataset:size()
	 return torch.round(data.data:size(1)/params.seq_length-0.5)
      end

      setmetatable(dataset, {__index =
				function(self, index)
				   return {data.data:narrow(1, (index-1)*params.seq_length+1, params.seq_length),
					   data.data:narrow(1, (index-1)*params.seq_length+2, params.seq_length)}
      end})
   else -- mlp train
      function dataset:size()
	 return torch.round(data.data:size(1)/params.batch_size-0.5)
      end
      setmetatable(dataset, {__index =
				function(self, index)
				   return {
				      data.data:narrow(1, (index-1)*params.batch_size+1, params.batch_size),
				      label.data:narrow(1, (index-1)*params.batch_size+1, params.batch_size)}
      end})
   end

   return dataset
end
   
local function valid_run(valid_dataset, model, params)
   local len = valid_dataset:size()
   -- disable dropout
   if(model.model_type == 'lstm') then
      g_disable_dropout(model.rnns)
   else
      g_disable_dropout(model.core_network)
   end

   local num = len
   if num>250 then
      num = 250
   end
   local perps = torch.zeros(num)
   
   for i=1, num do
      local idx = math.random(len)%len +1
      local perp = fp(valid_dataset, idx, model, params)
      perps[i] = perp
   end
   -- enable dropout
   if(model.model_type == 'lstm') then
      g_enable_dropout(model.rnns)
   else
      g_enable_dropout(model.core_network)
   end
   
   perps = torch.exp(perps:mean())
   local info = 'run valid perplexity: '..g_f3(perps)
   return info
end    

local function updateParameters(weights, gradWeights, model, params)
   model.norm_dw = 0
   for i=1, #gradWeights do
      local norm_dw = gradWeights[i]:norm()
      if norm_dw > params.max_grad_norm then
	 local shrink_factor = params.max_grad_norm / norm_dw
	 gradWeights[i]:mul(shrink_factor)
      end
      model.norm_dw = model.norm_dw + gradWeights[i]:norm()
      weights[i]:add(-params.lr, gradWeights[i])
   end
   model.norm_dw = model.norm_dw / #gradWeights
end

local function threadedTrain(model, data, label, valid_data, valid_label, params, cp_params)

   -- corner case: we are here to do batches
   -- no bach, no threading
   if params.batch_size == 1 then
      print('! WARNING: no batch => no thread')
      params.nThreads = 1
   end

   if params.nThreads == 1 then
      print('! WARNING: if you use no thread, better not use a thread-aware code [overheads ahead]')
   end

   print('multi-thread train begin')

   Threads.serialization('threads.sharedserialize')
   
   local threads = Threads(
      params.nThreads,
      function()
         require 'nn'
	 require 'nngraph'
	 require 'base'
      end,

      function()
	 print('thread '..__threadid..' start!')
         local model = model_clone(model, params)
	 local _, dweights = model.core_network:parameters()
         local data = data
         local label = label

	 local dataset = build_dataset(model, data, label, params)

	 function gupdate(idx)
	    local err = fp(dataset, idx, model, params)
	    --print(err..'  thread '..__threadid)
	    bp(dataset, idx, model, params)
	    return err, dweights
	 end
      end
   )

   local weights, _ = model.core_network:parameters()
   local dataset = build_dataset(model, data, label, params)
   local valid_dataset = build_dataset(model, valid_data, valid_label, params)

   params.lr = cp_params.lr
   local idx = cp_params.pos
   local step = cp_params.step
   local epoch = cp_params.epoch
   local total_cases = cp_params.total_cases
   local perps= cp_params.perps
   local tics = cp_params.tics
   local beginning_time = os.time() - tics
   print('Starting training')
   local epoch_size = dataset:size()
   local words_per_step
   if model.model_type == 'lstm' then
      words_per_step = params.batch_size * params.seq_length
   else
      words_per_step = params.batch_size * params.input_size
   end

   if perps == nil then
      perps = torch.zeros(epoch_size)
   end
   
   local log = torch.DiskFile(params.log_path, 'rw')
   while epoch < params.max_max_epoch do
      
      threads:addjob(
	 function(idx)
	    return gupdate(idx)
	 end,

	 function(err, dweights)
	    perps[idx] = err
	    updateParameters(weights, dweights, model, params)
	    total_cases = total_cases + words_per_step
	 end,
	 idx
      )

      idx = idx % dataset:size() + 1
      
      step = step + 1
      epoch = step / epoch_size
      if step % 50 == 0 then
	 threads:synchronize() -- accumulate grad
	 local spend_time = os.time() - beginning_time
	 local wps = torch.floor(total_cases / spend_time)
	 local since_beginning = g_d(spend_time / 60)
	 local info = 'epoch = ' .. g_f3(epoch) ..
	    ', train perp. = ' .. g_f3(torch.exp(perps:sum()/math.min(epoch_size, step))) ..
	    ', wps = ' .. wps ..
	    ', dw:norm() = ' .. g_f3(model.norm_dw) ..
	    ', lr = ' ..  g_f3(params.lr) ..
	    ', since beginning = ' .. since_beginning .. ' mins.'
	 print(info)
	 log:writeString(info..'\n')
	 -- save model to model_path file
	 cp(idx, params.lr, step, epoch, total_cases, perps, spend_time, weights, params.cp_path)
	 
      end
      
      if step % epoch_size ==0 then
	 if epoch > params.max_epoch and torch.round(epoch) % params.max_epoch == 0 then
	    params.lr = params.lr / params.decay
	 end
	 local info = valid_run(valid_dataset, model, params)
	 print(info)
	 log:writeString(info..'\n')
      end
      
      if step % 33 == 0 then
	 --cutorch.synchronize()
	 collectgarbage()
      end
   end

   threads:terminate()
   print("Training is over.")
   log:writeString('Training is over\n')
   log:close()
end

return threadedTrain
