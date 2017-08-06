--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

require('nngraph')
require('base')
local imdb = require('data')

-- Train 1 day and gives 82 perplexity.
--[[
local params = {batch_size=20,
                seq_length=35,
                layers=2,
                decay=1.15,
                rnn_size=1500,
                dropout=0.65,
                init_weight=0.04,
                lr=1,
                vocab_size=10000,
                max_epoch=14,
                max_max_epoch=55,
                max_grad_norm=10}
               ]]--

-- Trains 1h and gives test 115 perplexity.
-- lstm parameters
local lstm_params = {batch_size=20,
                seq_length=20,
                layers=2,
                decay=2,
                rnn_size=200,
                dropout=0.5,
                init_weight=0.1,
                lr=1,
                vocab_size=10000,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=5,
		cp_path='lstm_params.t7',
		log_path='perps.log'}

-- lstm model parameters
local lstm_model = {paramx,
		    paramdx, -- model parameters
		    s,       -- the state, lstm_params.seq_length x 2*layers x batch_size x rnn_size
		    ds,      -- the ds
		    start_s  -- the start state of s
}

-- lstm checkpoint parameters
local lstm_cp_params = {pos = 1,
			lr = lstm_params.lr,
			step = 0,
			epoch = 0,
			perps,
			total_cases = 0,
			tics = 0,
			paramx,
			paramdx}

-- multi-layer perception parameters
local mlp_params = {
                layers=4,
                decay=2,
                dropout=0.5,
                init_weight=0.1,
                lr=1,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=5,
		class_num = 10,
		cp_path='mlp_params.t7',
		log_path='mlp_loss.log'}

-- mlp model parameters
local mlp_model = {paramx,
		   paramdx,
}

-- mlp checkpoint parameters
local mlp_cp_params = {pos = 1,
		       lr = mlp_params.lr,
		       step = 0,
		       epoch = 0,
		       perps,
		       total_cases = 0,
		       tics = 0,
		       paramx,
		       paramdx}



local function transfer_data(x)
  return x --:cuda()
end

-- lstm unit
local function lstm_unit(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(lstm_params.rnn_size, 4*lstm_params.rnn_size)(x)
  local h2h = nn.Linear(lstm_params.rnn_size, 4*lstm_params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,lstm_params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

-- construct lstm model
local function lstm_module()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = nn.LookupTable(lstm_params.vocab_size,
                                                    lstm_params.rnn_size)(x)}
  local next_s           = {}
  local split         = {prev_s:split(2 * lstm_params.layers)}
  for layer_idx = 1, lstm_params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(lstm_params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm_unit(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(lstm_params.rnn_size, lstm_params.vocab_size)
  local dropped          = nn.Dropout(lstm_params.dropout)(i[lstm_params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s)})
  module:getParameters():uniform(-lstm_params.init_weight, lstm_params.init_weight)
  return transfer_data(module)
end

local function lstm_setup()
  print("Creating a RNN LSTM network.")
  local core_network = lstm_module()
  lstm_model.paramx, lstm_model.paramdx = core_network:getParameters()
  lstm_model.s = {}
  lstm_model.ds = {}
  lstm_model.start_s = {}
  for j = 0, lstm_params.seq_length do
    lstm_model.s[j] = {}
    for d = 1, 2 * lstm_params.layers do
      lstm_model.s[j][d] = transfer_data(torch.zeros(lstm_params.batch_size, lstm_params.rnn_size))
    end
  end
  for d = 1, 2 * lstm_params.layers do
    lstm_model.start_s[d] = transfer_data(torch.zeros(lstm_params.batch_size, lstm_params.rnn_size))
    lstm_model.ds[d] = transfer_data(torch.zeros(lstm_params.batch_size, lstm_params.rnn_size))
  end
  lstm_model.core_network = core_network
  lstm_model.rnns = g_cloneManyTimes(core_network, lstm_params.seq_length)
  lstm_model.norm_dw = 0
  lstm_model.err = transfer_data(torch.zeros(lstm_params.seq_length))
end

local function lstm_reset_state(state)
  state.pos = 1
  if lstm_model ~= nil and lstm_model.start_s ~= nil then
    for d = 1, 2 * lstm_params.layers do
      model.start_s[d]:zero()
    end
  end
end

local function lstm_reset_ds()
  for d = 1, #lstm_model.ds do
    lstm_model.ds[d]:zero()
  end
end

local function lstm_fp(state)
  g_replace_table(lstm_model.s[0], lstm_model.start_s)
  if state.pos + lstm_params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  for i = 1, lstm_params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = lstm_model.s[i - 1]
    lstm_model.err[i], lstm_model.s[i] = unpack(lstm_model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end
  g_replace_table(lstm_model.start_s, lstm_model.s[lstm_params.seq_length])
  return lstm_model.err:mean()
end

local function lstm_bp(state)
  lstm_model.paramdx:zero()
  lstm_reset_ds()
  for i = lstm_params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = lstm_model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local tmp = lstm_model.rnns[i]:backward({x, y, s},
                                       {derr, lstm_model.ds})[3]
    g_replace_table(model.ds, tmp)
    --cutorch.synchronize()
  end
  state.pos = state.pos + lstm_params.seq_length
  lstm_model.norm_dw = lstm_model.paramdx:norm()
  if lstm_model.norm_dw > lstm_params.max_grad_norm then
    local shrink_factor = lstm_params.max_grad_norm / lstm_model.norm_dw
    lstm_model.paramdx:mul(shrink_factor)
  end
  lstm_model.paramx:add(lstm_model.paramdx:mul(-lstm_params.lr))
end

-- resume lstm model params from checkpoint
local function resume_lstm_cp()
   lstm_cp_params = torch.load(lstm_params.cp_path)
   lstm_model.paramx:clone(lstm_cp_params.paramx)
   lstm_model.paramdx:clone(lstm_cp_params.paramdx)
end

-- lstm checkpoint
local function lstm_cp(pos, lr, step, epoch, total_cases, perps, tics, paramx, paramdx)
   lstm_cp_params.pos = pos
   lstm_cp_params.lr = lr
   lstm_cp_params.step = step
   lstm_cp_params.epoch = epoch
   lstm_cp_params.total_cases = total_cases
   lstm_cp_params.perps = perps
   lstm_cp_params.tics = tics
   lstm_cp_params.paramx = paramx
   lstm_cp_params.paramdx = paramdx
   torch.save(lstm_params.cp_path, lstm_cp_params)
end

local function load_lstm_network(state)
   lstm_reset_state(state)
   lstm_setup()
   
    -- check point if saved
   local file = io.open(lstm_params.cp_path, "rb")
   if file then -- file exist
      file:close()
      resume_lstm_cp()
   else
      print('faild to find saved lstm model, cannt use!')
      os.exit()
   end
   
end

-- test lstm model, output perplexity
local function lstm_test(state_test)

   load_lstm_network(state_test)

   g_disable_dropout(lstm_model.rnns)
   
   local perp = 0
   local len = state_test.data:size(1)
   g_replace_table(model.s[0], model.start_s)
   for i = 1, (len - 1) do
      local x = state_test.data[i]
      local y = state_test.data[i + 1]
      perp_tmp, model.s[1] = unpack(lstm_model.rnns[1]:forward({x, y, lstm_model.s[0]}))
      perp = perp + perp_tmp[1]
      g_replace_table(lstm_model.s[0], lstm_model.s[1])
   end
   local info = "Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1)))
   print(info)
   g_enable_dropout(lstm_model.rnns)
   return info
end

local function lstm_train(state_train, state_test, retrain)
  g_init_cpu(arg)

  lstm_reset_state(state_train)
  lstm_setup()

  -- check point if saved
  local file = io.open(lstm_params.cp_path, "rb")
  if file then -- file exist
     file:close()
     if retrain == false then -- don't train from scratch
	resume_lstm_cp()
     end
  end

  state_train.pos = lstm_cp_params.pos
  lstm_params.lr = lstm_cp_params.lr
  local step = lstm_cp_params.step
  local epoch = lstm_cp_params.epoch
  local total_cases = lstm_cp_params.total_cases
  local perps= lstm_cp_params.perps
  local tics = lstm_cp_params.tics
  local beginning_time = torch.tic() - tics
  local start_time = torch.tic() - tics
  print('Starting training')
  local words_per_step = lstm_params.seq_length * lstm_params.batch_size
  local epoch_size = torch.floor(state_train.data:size(1) / lstm_params.seq_length)

  log = torch.DiskFile(lstm_params.log_path, 'rw')
  while epoch < lstm_params.max_max_epoch do
    local perp = lstm_fp(state_train)
    if perps == nil then
      perps = torch.zeros(epoch_size):add(perp)
    end
    perps[step % epoch_size + 1] = perp
    step = step + 1
    lstm_bp(state_train)
    total_cases = total_cases + words_per_step
    epoch = step / epoch_size
    if step % torch.round(epoch_size / 10) == 10 then
      local wps = torch.floor(total_cases / torch.toc(start_time))
      local since_beginning = g_d(torch.toc(beginning_time) / 60)
      local info = 'epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
            ', wps = ' .. wps ..
            ', dw:norm() = ' .. g_f3(model.norm_dw) ..
            ', lr = ' ..  g_f3(lstm_params.lr) ..
	 ', since beginning = ' .. since_beginning .. ' mins.'
      print(info)
      log:writeString(info..'\n')
       -- save model to model_path file
      lstm_cp(state_train.pos, lstm_params.lr, step, epoch, total_cases, perps, torch.tic() - beginning_time, lstm_model.paramx, lstm_model.paramdx)
    end

    if step % epoch_size == 0 then
       info = lstm_test(state_test) -- test as valid
       log:writeString(info..'\n')
      if epoch > lstm_params.max_epoch then
          lstm_params.lr = lstm_params.lr / lstm_params.decay
      end
    end
    if step % 33 == 0 then
      --cutorch.synchronize()
      collectgarbage()
    end
  end
  
  print("Training is over.")
  log:writeString('Training is over\n')
  log:close()
end

-------------------------------cutting line----------------------------------------------------------------
   
-- input from lstm, h1+h2+...+ht

local function mlp_input(state)

   data = state.data[state.pos]
   local len = data:size(1)
   x_input = {}
   g_replace_table(lstm_model.s[0], lstm_model.start_s)
   for i = 1, (len-1)  do
      local x = data[i]
      local y = data[i + 1]
      perp_tmp, lstm_model.s[1] = unpack(lstm_model.rnns[1]:forward({x, y, lstm_model.s[0]}))
      g_replace_table(lstm_model.s[0], lstm_model.s[1])
      table.insert(x_input, lstm_model.s[-1])
   end
   
   result = nn.CAddTable(x_input)/#x_input
   return result
end

-- construct mlp model
local function mlp_module()
   
   local x = nn.Identity()()
   local y = nn.Identity()()
   local i = {[0] = x}
   
   local del_size = torch.floor(lstm_params.rnn_size/mlp_params.layers)
   
   for layer_idx = 1, mlp_params.layers-1 do
      local dropped = nn.Dropout()(i[layer_idx - 1])
      local prev_h_size = lstm_params.rnn_size-(layer_idx-1)*del_size
      local next_h_size = prev_h_size - del_size
      local next_h = nn.Linear(prev_h_size, next_h_size)(dropped)
      i[layer_idx] = next_h
   end

   local prev_h_size = lstm_params.rnn_size-(mlp_params.layers-1)*del_size
   local dropped = nn.Dropout(mlp_params.dropout)(i[mlp_params.layers-1])
   local h2y = nn.Linear(prev_h_size, mlp_params.class_num)(dropped)
   
   local pred = nn.LogSoftMax(h2y)
   local err = nn.ClassNLLCriterion()({pred, y})
   local module = nn.gModule({x, y}, {err})
   module:getParameters():uniform(-mlp_params.init_weight, mlp_params.init_weight)
   return module
end

local function mlp_setup()
   print("Creating a Multi-Layer Perception network.")
   local core_network = mlp_module()
   mlp_model.paramx, mlp_model.paramdx = core_network:getParameters()
   mlp_model.core_network = core_network
   model.norm_dw = 0
   model.err = 0
end

local function mlp_fp(state, label)

   local len = #state.data
   for i=1, len do
      if state.pos > len then -- reset
	 state.pos = 1
      end
      if label.data[state.pos] == 0 then -- unmark data, skip
	 state.pos = state.pos + 1
      else
	 break
      end
   end   

   local x = mlp_input(state)
   local y = label.data[state.pos]
   local loss = model.core_network:forward({x, y})
   return loss
end

local function mlp_bp(state, label)
   mlp_model.paramdx:zero()
   x = mlp_input(state)
   y=label.data[state.pos]
   mlp_model.core_network:backward({x, y}, {torch.ones(1)})

   mlp_model.norm_dw = mlp_model.paramdx:norm()
   if mlp_model.norm_dw > mlp_params.max_grad_norm then
      local shrink_factor = mlp_params.max_grad_norm / mlp_model.norm_dw
      mlp_model.paramdx:mul(shrink_factor)
   end
   mlp_model.paramx:add(mlp_model.paramdx:mul(mlp_params.lr))
   state.pos = state.pos + 1
end

local function resume_mlp_cp()
   mlp_cp_params = torch.load(mlp_params.cp_path)
   mlp_model.paramx:clone(mlp_cp_params.paramx)
   mlp_model.paramdx:clone(mlp_cp_params.paramdx)
end

-- lstm checkpoint
local function mlp_cp(pos, lr, step, epoch, total_cases, perps, tics, paramx, paramdx)
   mlp_cp_params.pos = pos
   mlp_cp_params.lr = lr
   mlp_cp_params.step = step
   mlp_cp_params.epoch = epoch
   mlp_cp_params.total_cases = total_cases
   mlp_cp_params.perps = perps
   mlp_cp_params.tics = tics
   mlp_cp_params.paramx = paramx
   mlp_cp_params.paramdx = paramdx
   torch.save(mlp_params.cp_path, mlp_cp_params)
end

local function mlp_train(state_train, label_train, state_test, label_test, retrain)
   g_init_cpu(arg)

   -- create lstm network
   load_lstm_network()
   g_disable_dropout(lstm_model.rnns)

   state_train.pos = 1
   mlp_setup()

   -- check point if saved
   local file = io.open(mlp_params.cp_path, "rb")
   if file then -- file exist
     file:close()
     if retrain == false then -- don't train from scratch
	resume_mlp_cp()
     end
  end

  state_train.pos = mlp_cp_params.pos
  mlp_params.lr = mlp_cp_params.lr
  local step = mlp_cp_params.step
  local epoch = mlp_cp_params.epoch
  local total_cases = mlp_cp_params.total_cases
  local perps= mlp_cp_params.perps
  local tics = mlp_cp_params.tics
  local beginning_time = torch.tic() - tics
  local start_time = torch.tic() - tics
  local total_cases = 0
  print('Starting training')
  local epoch_size = #state_train.data

  log = torch.DiskFile(mlp_params.log_path, 'rw')
  while epoch < mlp_params.max_max_epoch do
    local perp = mlp_fp(state_train, label_train)
    if perps == nil then
      perps = torch.zeros(epoch_size):add(perp)
    end
    perps[step % epoch_size + 1] = perp
    step = step + 1
    total_cases = total_cases + #state_train.data[state_train.pos]
    mlp_bp(state_train, label_train)
    epoch = step / epoch_size
    if step % torch.round(epoch_size / 10) == 10 then
      local wps = torch.floor(total_cases / torch.toc(start_time))
      local since_beginning = g_d(torch.toc(beginning_time) / 60)
      info = 'epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
            ', wps = ' .. wps ..
            ', dw:norm() = ' .. g_f3(mlp_model.norm_dw) ..
            ', lr = ' ..  g_f3(mlp_params.lr) ..
	 ', since beginning = ' .. since_beginning .. ' mins.'
      print(info)
      log:writeString(info..'\n')
       -- save model to model_path file
      mlp_cp(state_train.pos, mlp_params.lr, step, epoch, total_cases, perps, torch.tic() - beginning_time, mlp_model.paramx, mlp_model.paramdx)
    end

    if step % epoch_size == 0 then
       info = mlp_test(state_test, label_test) -- test data for valid
       log:writeString(info..'\n')
      if epoch > lstm_params.max_epoch then
          mlp_params.lr = mlp_params.lr / mlp_params.decay
      end
    end
    if step % 33 == 0 then
      --cutorch.synchronize()
      collectgarbage()
    end
  end
  
  print("Training is over.")
  log:writeString('Training is over\n')
  log:close()

  g_enable_dropout(lstm_model.rnns)
end

local function mlp_test(state_test, state_label)
   load_lstm_network()
   g_disable_dropout(lstm_model.rnns)

   state_test.pos = 1
   mlp_setup()

  -- check point if saved
  local file = io.open(mlp_params.cp_path, "rb")
  if file then -- file exist
     file:close()
     resume_mlp_cp()
  else
     print('failed to find mlp model, cannt test!')
     os.exit()
  end

   g_disable_dropout(mlp_model.core_network)
   local perps=0
   for i=1, #state_test.data do
      perps = perps + mlp_fp(state_test, state_label)
   end
   g_enable_dropout(mlp_model.core_network)

   local result = torch.exp(perps/#state_test.data)

   local info = 'mlp test perplexity : '..g_f3(result)
   print(info)
   return info
end
      

local function main()

   local train_data_path = './data/aclImdb/train-merge/train.data'
   local train_label_path = './data/aclImdb/train-merge/train.label'
   local test_data_path = './data/aclImdb/test-merge/test.data'
   local test_label_path = './data/aclImdb/test-merge/test.label'

   -- create vocab dictioanry before data use
   imdb.build_data_vocab(train_data_path)
   lstm_params.vocab_size = imdb.get_vocab_size()
   
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('LSTM Sentiment Classification')
   cmd:text('Example:')
   cmd:text('$> th main.lua --lstm_mode 1 --mlp_mode 1')
   cmd:text('Options:')
   cmd:option('--lstm_mode', -1, '0:test,1:train,2:train from scratch')
   cmd:option('--mlp_mode', -1, '0:test,1:train,2:train from scratch')
   cmd:text()

   -- parse parameters
   opt = cmd:parse(arg or {})

   -- parse lstm mode
   if(opt.lstm_mode == 0) then
      print('test lstm model')
      lstm_test_state = {data=imdb.lstm_load_data(test_data_path, lstm_params.batch_size)}
      lstm_test(lstm_test_state)
   end  
   if opt.lstm_mode == 1 then
      print('train lstm model')
      lstm_train_state = {data=imdb.lstm_load_data(train_data_path, lstm_params.batch_size)}
      lstm_test_state = {data=imdb.lstm_load_data(test_data_path, lstm_params.batch_size)}
      lstm_train(lstm_train_state, lstm_test_state, false)
   end
   if opt.lstm_mode == 2 then
      print('train lstm model from scratch')
      lstm_train_state = {data=imdb.lstm_load_data(train_data_path, lstm_params.batch_size)}
      lstm_test_state = {data=imdb.lstm_load_data(test_data_path, lstm_params.batch_size)}
      lstm_train(lstm_train_state,lstm_test_state, true)
   end

   -- parse mlp mode
   if opt.mlp_mode == 0 then
      print('test mlp classifier')
      mlp_test_state = {data=imdb.mlp_load_data(test_data_path)}
      mlp_test_label = {data=imdb.mlp_load_label(test_label_path)}
      assert(#mlp_test_state == #mlp_test_label) -- data number should equal label number
      mlp_test(mlp_test_state, mlp_test_label)
   end
   if(opt.mlp_mode == 1) then
      print('train mlp model')
      mlp_train_state = {data=imdb.mlp_load_data(train_data_path)}
      mlp_train_label = {data=imdb.mlp_load_label(train_label_path)}
      assert(#mlp_train_state == #mlp_train_label) -- data number should equal label number

      mlp_test_state = {data=imdb.mlp_load_data(test_data_path)}
      mlp_test_label = {data=imdb.mlp_load_label(test_label_path)}
      assert(#mlp_test_state == #mlp_test_label) -- data number should equal label number
      
      mlp_train(mlp_train_state, mlp_train_label, mlp_test_state, mlp_test_label, false)
   end
   if opt.mlp_mode == 2 then
      print('train the mlp model from scratch')
      mlp_train_state = {data=imdb.mlp_load_data(train_data_path)}
      mlp_train_label = {data=imdb.mlp_load_label(train_label_path)}
      assert(#mlp_train_state == #mlp_train_label) -- data number should equal label number

      mlp_test_state = {data=imdb.mlp_load_data(test_data_path)}
      mlp_test_label = {data=imdb.mlp_load_label(test_label_path)}
      assert(#mlp_test_state == #mlp_test_label) -- data number should equal label number
      
      mlp_train(mlp_train_state, mlp_train_label, mlp_test_state, mlp_test_label, true)
   end
end

main()
