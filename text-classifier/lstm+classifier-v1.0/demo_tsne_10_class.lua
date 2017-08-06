local manifold = require 'manifold'
require 'gnuplot'
require 'image'
local stringx = require('pl.stringx')
local file = require('pl.file')

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

-- function to show an MNIST 2D group scatter plot:
local function show_scatter_plot(method, mapped_x, labels, fname)

   -- count label sizes:
   local K = 10
   local cnts = torch.zeros(K)
   for n = 1,labels:nElement() do
      cnts[labels[n]] = cnts[labels[n]] + 1
   end

   -- separate mapped data per label:
   mapped_data = {}
   for k = 1,K do
      if cnts[k] ~= 0 then
	 mapped_data[k] = {'Class ' .. k, torch.Tensor(cnts[k], opts.ndims), '+'}
      end
   end
   local offset = torch.Tensor(K):fill(1)
   for n = 1,labels:nElement() do
      mapped_data[labels[n]][2][offset[labels[n]]]:copy(mapped_x[n])
      offset[labels[n]] = offset[labels[n]] + 1
   end

   local new_mapped_data = {}
   for k=1, K do
      if cnts[k] ~= 0 then
	 table.insert(new_mapped_data, mapped_data[k])
      end
   end

   -- show results in scatter plot:
   gnuplot.pdffigure(fname); gnuplot.grid(true); gnuplot.title(method)
   gnuplot.plot(new_mapped_data)
   gnuplot.plotflush()
end


-- show map with original digits:
local function show_map(method, mapped_data, X)

   -- draw map with original digits:
   local im_size = 2048
   local background = 0
   local background_removal = true
   map_im = manifold.draw_image_map(mapped_data, X:resize(X:size(1), 1, 28, 28), im_size, background, background_removal)

   -- plot results:
   image.display{image=map_im, legend=method, zoom=0.5}
   image.savePNG(method .. '.png', map_im)
end


-- function that performs demo of t-SNE code on MNIST:
local function demo_tsne(data_path, label_path)

   local x = torch.load(data_path)
   local labels = mlp_load_label(label_path)
   
   -- amount of data to use for test:
   local N = labels:size(1)

   -- load subset of MNIST test data:
   --local mnist = require 'mnist'
   --local testset = mnist.testdataset()
   local testset = {data=x, label=labels}
   testset.size  = N
   testset.data  = testset.data:narrow(1, 1, N)
   testset.label = testset.label:narrow(1, 1, N)
   local x = torch.DoubleTensor(testset.data:size()):copy(testset.data)
   --x:resize(x:size(1), x:size(2) * x:size(3))
   local labels = testset.label
   
   
   -- run t-SNE:
   local timer = torch.Timer()
   opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = false}
   mapped_x1 = manifold.embedding.tsne(x, opts)
   print('Successfully performed t-SNE in ' .. timer:time().real .. ' seconds.')
   show_scatter_plot('t-SNE', mapped_x1, labels,'t-SNE.pdf')
   --show_map('t-SNE', mapped_x1, x:clone())

   -- run Barnes-Hut t-SNE:
   opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = true, theta = 0.5}
   timer:reset()
   mapped_x2 = manifold.embedding.tsne(x, opts)
   print('Successfully performed Barnes Hut t-SNE in ' .. timer:time().real .. ' seconds.')
   show_scatter_plot('Barnes-Hut t-SNE', mapped_x2, labels, 'Barnes-Hut_t-SNE.pdf')
   --show_map('Barnes-Hut t-SNE', mapped_x2, x:clone())
end


-- run the demo:
local data_path = './bin/mlp_train.data'
local label_path = './data/aclImdb-test/train-merge/train.label'
demo_tsne(data_path, label_path)

local data_path = './bin/mlp_test.data'
local label_path = './data/aclImdb-test/test-merge/test.label'
--demo_tsne(data_path, label_path)
