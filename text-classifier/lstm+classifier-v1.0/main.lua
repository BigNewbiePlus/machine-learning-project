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
local unpack = unpack or table.unpack
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
		nThreads = 1,
		cp_path='./bin/lstm_params.t7',
		log_path='./bin/lstm_perps.log'}


-- multi-layer perception parameters
local mlp_params = {batch_size=20,
                layers=4,
                decay=1.15,
		input_size = lstm_params.rnn_size,
                dropout=0.5,
                init_weight=0.1,
                lr=0.5,
                max_epoch=40,
                max_max_epoch=1300,
                max_grad_norm=5,
		class_num = 10,
		nThreads = 1,
		cp_path='./bin/mlp_params.t7',
		log_path='./bin/mlp_perps.log',
		mlp_train_data_path='./bin/mlp_train.data',
		mlp_test_data_path='./bin/mlp_test.data'}


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
  -- weights initialize
  local weights, _ = module:parameters()
  for i=1, #weights do
     weights[i]:uniform(-lstm_params.init_weight, lstm_params.init_weight)
  end
  return transfer_data(module)
end

-- lstm setup for model and checkpoint
local function lstm_setup()
   local lstm_model = {}
   local lstm_cp = {}
   print("Creating a RNN LSTM network.")
   local core_network = lstm_module()
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
   lstm_model.model_type = 'lstm'

   
   -- lstm checkpoint initialize
   lstm_cp.pos = 1
   lstm_cp.lr = lstm_params.lr
   lstm_cp.step = 1
   lstm_cp.epoch = 0
   lstm_cp.total_cases = 0
   lstm_cp.tics = 0

   return lstm_model, lstm_cp
end

-- resume model params from checkpoint
local function resume_cp(model, cp_path)
   local cp_params = torch.load(cp_path)
   local weights, _ = model.core_network:parameters()
   for i=1, #weights do
      weights[i]:copy(cp_params.weights[i])
   end
   return cp_params
end

local function init_lstm(resume)
   local lstm_model, lstm_cp = lstm_setup()

   -- check point if saved
   local file = io.open(lstm_params.cp_path, "rb")
   if file then -- file exist
      file:close()
      if resume == true then -- load previous trained parameters
	 lstm_cp = resume_cp(lstm_model, lstm_params.cp_path)
      end
   else
      print('lstm no checkpoint')
   end
   return lstm_model,lstm_cp
end

-- test lstm model, output perplexity
local function lstm_test(state_test)

   local resume = true
   local lstm_model, _ = init_lstm(resume)

   g_disable_dropout(lstm_model.core_network)
   
   local perps = 0, perp
   local len = state_test.data:size(1)
   g_replace_table(lstm_model.s[0], lstm_model.start_s)
   local num_test = math.min(len, 250)
   for i = 1, num_test do
      local idx = math.random(num_test)%num_test+1
      local x = state_test.data[idx]
      local y = state_test.data[idx + 1]
      perp, lstm_model.s[1] = unpack(lstm_model.core_network:forward({x, y, lstm_model.s[0]}))
      perps = perps + perp[1]
      g_replace_table(lstm_model.s[0], lstm_model.s[1])
   end
   local info = "Test set perplexity : " .. g_f3(torch.exp(perps / num_test))
   print(info)
   g_enable_dropout(lstm_model.core_network)
   return info
end
   
local function lstm_train(state_train, state_test, retrain)
   g_init_cpu(arg)
   local resume = not retrain
   local lstm_model,lstm_cp = init_lstm(resume)
   local threadedtrain = require('threadedtrain')
   threadedtrain(lstm_model, state_train, nil, state_test, nil, lstm_params, lstm_cp)
end

-------------------------------cutting line----------------------------------------------------------------
   
-- convert raw data to mlp input data
local function mlp_input(raw_data, lstm_model)

   for d = 1, 2 * lstm_params.layers do
	 lstm_model.s[0][d]:zero()
      end
   
   local len = raw_data:size(1)
   local x_input = {}
   local index = lstm_params.layers * 2
   
   local x = torch.ones(1)
   local y = torch.ones(1)
   for i = 1, (len-1)  do
      x[1] = raw_data[i]
      y[1] = raw_data[i + 1]
      _, lstm_model.s[1] = unpack(lstm_model.core_network:forward({x, y, lstm_model.s[0]}))
      g_replace_table(lstm_model.s[0], lstm_model.s[1])
      local temp = torch.ones(mlp_params.input_size)
      table.insert(x_input, temp:copy(lstm_model.s[1][index]))
   end
   local mean_pooling = torch.zeros(mlp_params.input_size)
   for i=1, #x_input do
      mean_pooling:add(x_input[i])
   end
   mean_pooling:div(#x_input)
   return mean_pooling
end

local function mlp_data_preprocess(lstm_model, state, path, retrain)
   assert(lstm_params.rnn_size == mlp_params.input_size)

   -- if exists preprocess data
   local file = io.open(path, "rb")
   if file then -- file exist
      file:close()
      if not retrain then
	 local x = torch.load(path)
	 print('load from previous converted data')
	 return x
      end
   end
   print('convert data begin!')
   for j = 0, 1 do
      lstm_model.s[j] = {}
      for d = 1, 2 * lstm_params.layers do
	 lstm_model.s[j][d] = transfer_data(torch.zeros(lstm_params.rnn_size))
      end
   end
   
   local len = #state.data
   local step = torch.round(len*0.01)
   local x = torch.zeros(len, mlp_params.input_size)
   print('convert len:'..len)
   for i=1, len do
      local data = mlp_input(state.data[i], lstm_model)
      x[i] = data
      if i%step==0 then
	 print('convert percent: '..g_f3(i*100/len))
      end
   end
   torch.save(path, x)
   print('convert data end!')
   return x
end

-- construct mlp model
local function mlp_module()
   
   local x = nn.Identity()()
   local y = nn.Identity()()
   local i = {[0] = x}
   
   local del_size = torch.floor(mlp_params.input_size/mlp_params.layers)
   
   for layer_idx = 1, mlp_params.layers-1 do
      local dropped = nn.Dropout(mlp_params.dropout)(i[layer_idx - 1])
      local prev_h_size = mlp_params.input_size-(layer_idx-1)*del_size
      local next_h_size = prev_h_size - del_size
      local next_h = nn.Linear(prev_h_size, next_h_size)(dropped)
      i[layer_idx] = next_h
   end
   local prev_h_size = mlp_params.input_size-(mlp_params.layers-1)*del_size
   local dropped = nn.Dropout(mlp_params.dropout)(i[mlp_params.layers-1])
   local h2y = nn.Linear(prev_h_size, mlp_params.class_num)(dropped)
   local pred = nn.LogSoftMax()(h2y)
   local err = nn.ClassNLLCriterion()({pred, y})
   local module = nn.gModule({x, y}, {err})

   -- weights initialize
   local weights, _ = module:parameters()
   for i=1, #weights do
      weights[i]:uniform(-mlp_params.init_weight, mlp_params.init_weight)
   end
   return module
end

local function mlp_setup()
   local mlp_model = {}
   local mlp_cp = {}
   print("Creating a Multi-Layer Perception network.")
   local core_network = mlp_module()
   mlp_model.core_network = core_network
   mlp_model.norm_dw = 0
   mlp_model.err = 0
   mlp_model.model_type = 'mlp'

   mlp_cp.pos = 1
   mlp_cp.lr = mlp_params.lr
   mlp_cp.step = 1
   mlp_cp.epoch = 0
   mlp_cp.total_cases = 0
   mlp_cp.tics = 0
   return mlp_model, mlp_cp
end

local function init_mlp(resume)
   local mlp_model, mlp_cp = mlp_setup()
   -- check point if saved
   local file = io.open(mlp_params.cp_path, "rb")
   if file then -- file exist
      file:close()
      if resume == true then -- don't train from scratch
	 mlp_cp = resume_cp(mlp_model, mlp_params.cp_path)
      end
   else
      print('mlp no chekpoint')
   end
   return mlp_model, mlp_cp
end

   
local function mlp_train(state_train, label_train, state_test, label_test, retrain)
   g_init_cpu(arg)
   local resume = true
   local lstm_model,_ = init_lstm(resume)
   resume = not retrain
   local mlp_model, mlp_cp = init_mlp(resume)
   -- data pre process
   g_disable_dropout(lstm_model.core_network)
   local state_train = {data=mlp_data_preprocess(lstm_model, state_train, mlp_params.mlp_train_data_path, retrain)}
   local state_test = {data=mlp_data_preprocess(lstm_model, state_test, mlp_params.mlp_test_data_path, retrain)}
   threadedtrain = require('threadedtrain')
   threadedtrain(mlp_model, state_train, label_train, state_test, label_test, mlp_params, mlp_cp)
end

local function mlp_test(state_test, state_label)
   -- load lstm and mlp model
   local resume = true
   local lstm_model,_ = init_lstm(resume)
   local mlp_model, _ = init_mlp(resume)
   -- preprocess mlp data
   g_disable_dropout(lstm_model.core_network)
   g_disable_dropout(mlp_model.core_network)
   local state_test = {data=mlp_data_preprocess(lstm_model, state_test, mlp_params.mlp_test_data_path, false)}
   
   local perps=0
   local len = torch.round(state_test.data:size(1)/mlp_params.batch_size-0.5)
   for i=1, len do
      local x = state_test.data:narrow(1, (i-1)*mlp_params.batch_size+1, mlp_params.batch_size)
      local y = state_label.data:narrow(1, (i-1)*mlp_params.batch_size+1, mlp_params.batch_size)
      local perp = mlp_model.core_network:forward({x, y})
      perps = perps + perp[1]
   end
   g_enable_dropout(mlp_model.core_network)
   g_enable_dropout(lstm_model.core_network)

   local result = torch.exp(perps/len)

   local info = 'mlp test perplexity : '..g_f3(result)
   print(info)
   return info
end
      

local function main()

   local train_data_path = './data/aclImdb-test/train-merge/train.data'
   local train_label_path = './data/aclImdb-test/train-merge/train.label'
   local test_data_path = './data/aclImdb-test/test-merge/test.data'
   local test_label_path = './data/aclImdb-test/test-merge/test.label'

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
   local opt = cmd:parse(arg or {})

   -- parse lstm mode
   if(opt.lstm_mode == 0) then
      print('test lstm model')
      local lstm_test_state = {data=imdb.lstm_load_data(test_data_path, lstm_params.batch_size)}
      lstm_test(lstm_test_state)
   end  
   if opt.lstm_mode == 1 then
      print('train lstm model')
      local lstm_train_state = {data=imdb.lstm_load_data(train_data_path, lstm_params.batch_size)}
      local lstm_test_state = {data=imdb.lstm_load_data(test_data_path, lstm_params.batch_size)}
      lstm_train(lstm_train_state, lstm_test_state, false)
   end
   if opt.lstm_mode == 2 then
      print('train lstm model from scratch')
      local lstm_train_state = {data=imdb.lstm_load_data(train_data_path, lstm_params.batch_size)}
      local lstm_test_state = {data=imdb.lstm_load_data(test_data_path, lstm_params.batch_size)}
      lstm_train(lstm_train_state,lstm_test_state, true)
   end

   -- parse mlp mode
   if opt.mlp_mode == 0 then
      print('test mlp classifier')
      local mlp_test_state = {data=imdb.mlp_load_data(test_data_path)}
      local mlp_test_label = {data=imdb.mlp_load_label(test_label_path)}
      assert(#mlp_test_state.data == mlp_test_label.data:size(1)) -- data number should equal label number
      mlp_test(mlp_test_state, mlp_test_label)
   end
   if(opt.mlp_mode == 1) then
      print('train mlp model')
      local mlp_train_state = {data=imdb.mlp_load_data(train_data_path)}
      local mlp_train_label = {data=imdb.mlp_load_label(train_label_path)}

      assert(#mlp_train_state.data == mlp_train_label.data:size(1)) -- data number should equal label number

      local mlp_test_state = {data=imdb.mlp_load_data(test_data_path)}
      local mlp_test_label = {data=imdb.mlp_load_label(test_label_path)}
      assert(#mlp_test_state.data == mlp_test_label.data:size(1)) -- data number should equal label number
      
      mlp_train(mlp_train_state, mlp_train_label, mlp_test_state, mlp_test_label, false)
   end
   if opt.mlp_mode == 2 then
      print('train the mlp model from scratch')
      local mlp_train_state = {data=imdb.mlp_load_data(train_data_path)}
      local mlp_train_label = {data=imdb.mlp_load_label(train_label_path)}
      assert(#mlp_train_state.data == mlp_train_label.data:size(1)) -- data number should equal label number

      local mlp_test_state = {data=imdb.mlp_load_data(test_data_path)}
      local mlp_test_label = {data=imdb.mlp_load_label(test_label_path)}
      assert(#mlp_test_state.data == mlp_test_label.data:size(1)) -- data number should equal label number
      
      mlp_train(mlp_train_state, mlp_train_label, mlp_test_state, mlp_test_label, true)
   end
end

main()
