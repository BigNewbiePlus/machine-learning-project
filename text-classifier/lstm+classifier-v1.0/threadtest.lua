local Threads = require 'threads'

Threads.serialization('threads.sharedserialize')

local data = torch.ones(100)
local data={data=data}
local threads = Threads(
   5,
   function()
      local dataset = data
      function gupdate(idx)
	 dataset.data[idx]=idx
	 return dataset
      end
   end
)

local num=0
for idx=1,100 do
   threads:addjob(
      function(idx)
	 return gupdate(idx)
      end,
      function(dataset)
	 num = num+1
	 
      end,
      idx)
end
print('syn before:'..num)
threads:synchronize()
print('syn after:'..num)
for idx=1,100 do
   threads:addjob(
      function(idx)
	 return gupdate(idx)
      end,
      function(dataset)
	 num = num+1
	 
      end,
      idx)
end
print('syn before:'..num)
threads:synchronize()
print('syn after:'..num)
threads:terminate()

