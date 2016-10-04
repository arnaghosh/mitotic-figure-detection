require 'image'

local numOfImages_mit = 100;
local numOfImages_Nmit = 100;
local imageAll = torch.Tensor(numOfImages_mit+numOfImages_Nmit,3,101,101)
local labelAll = torch.Tensor(numOfImages_mit+numOfImages_Nmit)

classes = {'mitosis','non-mitosis'}

for i=1,numOfImages_mit do
	imageAll[i] = 	image.load('/mitosis/im'..i..'.jpg');
	labelAll[i] = 1;
end

for i=1,numOfImages_Nmit do
	imageAll[numOfImages_mit+i] = 	image.load('/not-mitosis/im'..i..'.jpg');
	labelAll[numOfImages_mit+i] = 2;
end

local labelsShuffle = torch.randperm((#labelAll)[1])

local portionTrain = 0.9 	--90% data is trained
local trSize = torch.floor(labelsShuffle:size(1)*portionTrain)
local teSize = labelsShuffle:size(1) - trSize

trainData = {
	data = torch.Tensor(trSize,3,101,101),
	labels = torch.Tensor(trSize),
	size = function() return trSize end
}

testData = {
	data = torch.Tensor(teSize,3,101,101),
	labels = torch.Tensor(teSize),
	size = function() return teSize end
}

for i=1,trSize do
	trainData.data[i] = imageAll[labelsShuffle[i][1]]:clone()
	trainData.labels[i] = labelAll[labelsShuffle[i]]
end

for i=trSize+1,trSize+teSize do
	testData.data[i-trSize] = imageAll[labelShuffle[i][1]]:clone()
	trainData.labels[i-trSize] = labelAll[labelsShuffle[i]]
end

local mean = {}
local std = {}

for i=1,3 do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

for i=1,3 do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

print(sys.COLORS.red ..  '==> preprocessing data: normalize all three channels locally')

for i=1,3 do
   local trainMean = trainData.data[{ {},i }]:mean()
   local trainStd = trainData.data[{ {},i }]:std()

   local testMean = testData.data[{ {},i }]:mean()
   local testStd = testData.data[{ {},i }]:std()

   print('training data, '..i..'-channel, mean: ' .. trainMean)
   print('training data, '..i..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..i..'-channel, mean: ' .. testMean)
   print('test data, '..i..'-channel, standard deviation: ' .. testStd)
end

-- visualtisation part may be deleted. check before final run
if opt.visualize then
   local first256Samples_y = trainData.data[{ {1,256},1 }]
   image.display{image=first256Samples_y, nrow=16, legend='Some training examples: Y channel'}
   local first256Samples_y = testData.data[{ {1,256},1 }]
   image.display{image=first256Samples_y, nrow=16, legend='Some testing examples: Y channel'}
end


return {
   trainData = trainData,
   testData = testData,
   mean = mean,
   std = std,
   classes = classes
}
