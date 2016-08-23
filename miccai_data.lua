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
	testData.data[i] = imageAll[labelShuffle[i][1]]:clone()
	trainData.labels[i] = labelAll[labelsShuffle[i]]
end