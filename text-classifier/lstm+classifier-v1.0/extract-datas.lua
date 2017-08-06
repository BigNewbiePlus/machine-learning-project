local stringx = require('pl.stringx')
local file = require('pl.file')

local function extract_data(data_path, label_path, want_labels, save_data, save_label)
   local data = file.read(data_path)
   local labels = file.read(label_path)
   local data = stringx.splitlines(data)
   local labels = stringx.splitlines(labels)

   local extract_data=''
   local extract_labels=''

   local percent = torch.round(#data * 0.01)

   assert(#data == #labels)
   for i =1, #data do
      if want_labels[tonumber(labels[i])] == 1 then
	 extract_data = extract_data..data[i]..'\n'
	 extract_labels = extract_labels..labels[i]..'\n'
      end
      if i%percent == 0 then
	 print(string.format('percent:%.3f',i/#data))
      end
   end

   file.write(save_data, extract_data)
   file.write(save_label, extract_labels)
end

local data_path = './data/aclImdb/train-merge/train.data'
local label_path = './data/aclImdb/train-merge/train.label'
local want_labels={0,1,0,0,0,0,1,0,0}
local save_data = './data/aclImdb-test/train-merge/train.data'
local save_label = './data/aclImdb-test/train-merge/train.label'
extract_data(data_path, label_path, want_labels, save_data, save_label)

local data_path = './data/aclImdb/test-merge/test.data'
local label_path = './data/aclImdb/test-merge/test.label'
local want_labels={0,1,0,0,0,0,1,0,0}
local save_data = './data/aclImdb-test/test-merge/test.data'
local save_label = './data/aclImdb-test/test-merge/test.label'
extract_data(data_path, label_path, want_labels, save_data, save_label)

