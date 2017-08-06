--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

local stringx = require('pl.stringx')
local file = require('pl.file')

local vocab_idx = 1
local vocab_map = {}
local vocab_count = {}
local frequency = 5 -- the minimum frequency of word, if lower, unknown(id=1)

-- Stacks replicated, shifted versions of x_inp
-- into a single matrix of size x_inp:size(1) x batch_size.
local function replicate(x_inp, batch_size)
   local s = x_inp:size(1)
   local batch = torch.floor(s/batch_size)
   local x = torch.zeros(batch, batch_size)
   for i = 1, batch_size do
      local start = (i-1)*batch + 1
     local finish = start + batch - 1
     x:sub(1, batch, i, i):copy(x_inp:sub(start, finish))
   end
   return x
end

local function build_data_vocab(fname)
   local data = file.read(fname)
   data = stringx.replace(data, '\n', ' ')
   data = stringx.split(data)
   
   print('load data success, data num: '..g_d(#data))

   -- build vocab count
   local count_num=0
   for i = 1, #data do
      if vocab_count[data[i]] == nil then
	 vocab_count[data[i]] = 1
	 count_num = count_num + 1
      else
	 vocab_count[data[i]] = vocab_count[data[i]] + 1
      end
   end

   print('build vocab count success! vocab count num: '..g_d(count_num))

   -- build vocab map
   for i=1, #data do
      if vocab_map[data[i]] == nil then
	 if vocab_count[data[i]] >= frequency then
	    vocab_idx = vocab_idx + 1
	    vocab_map[data[i]] = vocab_idx
	 else
	    vocab_map[data[i]] = 1 -- unknown word
	 end
      end
   end

   print('build vocab map success!, vocab size : '..g_d(vocab_idx))
	    
   -- clear data
   data = nil
   vocab_count = nil
   collectgarbage()
end   
   
local function load_data(fname)
   local data = file.read(fname)
   data = stringx.replace(data, '\n', ' ')
   data = stringx.split(data)
   print(string.format("Loading %s, size of data = %d", fname, #data))
   local x = torch.zeros(#data)
   for i = 1, #data do
      if vocab_map[data[i]] == nil then -- unknown word, set to 1(unknown)
	 x[i] = 1
      else
	 x[i] = vocab_map[data[i]]
      end
   end
   data=nil
   collectgarbage()
   return x
end

local function lstm_load_data(fname, batch_size)
   local x = load_data(fname)
   x = replicate(x, batch_size)
   return x
end

local function mlp_load_data(fname)
   local data = file.read(fname)
   data = stringx.splitlines(data)
   for i =1, #data do
      local split = stringx.split(data[i])
      data[i] = torch.zeros(#split)
      for j=1, #split do
	 if vocab_map[split[j]] == nil then -- unknown word
	    data[i][j] = 1
	 else
	    data[i][j] = vocab_map[split[j]]
	 end
      end
   end
   return data
end

local function mlp_load_label(fname)
   local label = file.read(fname)
   label = stringx.splitlines(label)
   local len = #label
   local x = torch.zeros(len)
   for i=1, len do
      x[i] = tonumber(label[i])
   end
   label=nil
   collectgarbage()
   return x
end

local function get_vocab_size()
   return vocab_idx
end

return {build_data_vocab = build_data_vocab,
	lstm_load_data=lstm_load_data,
	mlp_load_data=mlp_load_data,
	mlp_load_label=mlp_load_label,
	get_vocab_size=get_vocab_size}
