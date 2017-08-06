--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the Apache 2 license found in the
--  LICENSE file in the root directory of this source tree. 
--

function g_disable_dropout(node)
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(g_disable_dropout)
    end
    return
  end
  if string.match(node.__typename, "Dropout") then
    node.train = false
  end
end

function g_enable_dropout(node)
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(g_enable_dropout)
    end
    return
  end
  if string.match(node.__typename, "Dropout") then
    node.train = true
  end
end

function g_cloneManyTimes(net, T)
   local clones = {}
   for i=1, T do
      clones[i] = net:clone('weight', 'bias', 'gradWeight', 'gradBias')
   end
  return clones
end

function g_init_cpu(args)
  local gpuidx = args
  gpuidx = gpuidx[1] or 1
 -- print(string.format("Using %s-th gpu", gpuidx))
 -- cutorch.setDevice(gpuidx)
  g_make_deterministic(1)
end

function g_make_deterministic(seed)
  torch.manualSeed(seed)
--  cutorch.manualSeed(seed)
  --torch.zeros(1, 1):cuda():uniform()
end

function g_replace_table(to, from)
  assert(#to == #from)
  for i = 1, #to do
    to[i]:copy(from[i])
  end
end

function g_f3(f)
  return string.format("%.3f", f)
end

function g_d(f)
  return string.format("%d", torch.round(f))
end
