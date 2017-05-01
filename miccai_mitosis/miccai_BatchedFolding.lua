require 'image'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'

cutorch.setDevice(1)
batchSize = 256; --256 images of both mitosis and non-mitosis.
fileLocation = '../Training_Data/Gen_Dataset/mitosis/'
Folds = {'A03','A04','A05','A07','A10','A11','A12','A14','A15','A17','A18'}
local numOfImages_mit = torch.Tensor(#Folds)
local numOfImages_Nmit = torch.Tensor(#Folds)
local totalIm_mit = 0;
local totalIm_Nmit = 0;
for i=1,(#Folds) do
  local mfile = io.open('Filenames_Spr_few/'..Folds[i]..'_mit.txt')
  local nmfile = io.open('Filenames_Spr_few/'..Folds[i]..'_nmit.txt')
  numOfImages_mit[i] = 0;
  for _ in io.lines('Filenames_Spr_few/'..Folds[i]..'_mit.txt') do
  	numOfImages_mit[i] = numOfImages_mit[i]+1;
  	totalIm_mit = totalIm_mit +1;
  end
  numOfImages_Nmit[i] = 0;
  for _ in io.lines('Filenames_Spr_few/'..Folds[i]..'_nmit.txt') do
  	numOfImages_Nmit[i] = numOfImages_Nmit[i]+1;
  	totalIm_Nmit = totalIm_Nmit +1;
  end
end
print(totalIm_mit,totalIm_Nmit, totalIm_mit+totalIm_Nmit)
local trNumOfImages = torch.Tensor(#Folds);
local allTrImages = 0;
for f=1,(#Folds) do
	if numOfImages_mit[f]>numOfImages_Nmit[f] then
		trNumOfImages[f] = numOfImages_Nmit[f]
		numOfImages_mit[f] = numOfImages_Nmit[f]
	else
		trNumOfImages[f] = numOfImages_mit[f]
		numOfImages_Nmit[f] = numOfImages_mit[f]
	end
	allTrImages = allTrImages + 2*trNumOfImages[f];
end
totalIm_mit = allTrImages/2;
totalIm_Nmit = allTrImages/2;
print(allTrImages)
local AllImages_mit = torch.ByteTensor(totalIm_mit,3,101,101);
local AllImages_Nmit = torch.ByteTensor(totalIm_Nmit,3,101,101);
print("1")
--local trImages = torch.ByteTensor(2,allTrImages/2,3,101,101);
print("2")
--AllImages_mit = {};
--AllImages_Nmit={};
local FoldIndex_mit = torch.Tensor(totalIm_mit);
local FoldIndex_Nmit = torch.Tensor(totalIm_Nmit);
local count_mit = 0;
local count_Nmit = 0;
local overallCount = 0;
for i=1,(#Folds) do
  print(i,numOfImages_mit[i],numOfImages_Nmit[i])
  local mfile = io.open('Filenames_Spr_few/'..Folds[i]..'_mit.txt')
  local nmfile = io.open('Filenames_Spr_few/'..Folds[i]..'_nmit.txt')
  local tempc=0;
  for l in io.lines('Filenames_Spr_few/'..Folds[i]..'_mit.txt') do
  	if tempc==numOfImages_mit[i] then break end
  	count_mit = count_mit +1;
  	tempc = tempc+1;
  	overallCount = overallCount+1;
  	local img = image.load('../Fold_Train_few/'..Folds[i]..'/mitosis/'..l,3,'byte');
  	AllImages_mit[count_mit]:copy(img)
  	FoldIndex_mit[count_mit]=i;
  	--if count_mit%10000==0 then print("10k done") end
  end
  tempc=0;
  for l in io.lines('Filenames_Spr_few/'..Folds[i]..'_nmit.txt') do
  	if tempc==numOfImages_Nmit[i] then break end
  	count_Nmit = count_Nmit +1;
  	tempc = tempc+1;
  	overallCount = overallCount+1;
  	local img = image.load('../Fold_Train_few/'..Folds[i]..'/not_mitosis/'..l,3,'byte');
  	AllImages_Nmit[count_Nmit]:copy(img);
  	FoldIndex_Nmit[count_Nmit]=i;
  end
  collectgarbage()
end
--------------------------------------------Training image names loaded above. Next validate/test image names to be loaded.------------------------

local VnumOfImages_mit = torch.Tensor(#Folds)
local VnumOfImages_Nmit = torch.Tensor(#Folds)
local VtotalIm_mit = 0;
local VtotalIm_Nmit = 0;
for i=1,(#Folds) do
  local mfile = io.open('Filenames/'..Folds[i]..'_mit.txt')
  local nmfile = io.open('Filenames/'..Folds[i]..'_nmit.txt')
  VnumOfImages_mit[i] = 0;
  for _ in io.lines('Filenames/'..Folds[i]..'_mit.txt') do
  	VnumOfImages_mit[i] = VnumOfImages_mit[i]+1;
  	VtotalIm_mit = VtotalIm_mit +1;
  end
  VnumOfImages_Nmit[i] = 0;
  for _ in io.lines('Filenames/'..Folds[i]..'_nmit.txt') do
  	VnumOfImages_Nmit[i] = VnumOfImages_Nmit[i]+1;
  	VtotalIm_Nmit = VtotalIm_Nmit +1;
  end
end

local VtrNumOfImages = torch.Tensor(#Folds);
local VallTrImages = 0;
for f=1,(#Folds) do
	VallTrImages = VnumOfImages_mit[f] + VnumOfImages_Nmit[f];
end

--local VAllImages_mit = torch.ByteTensor(VtotalIm_mit,3,101,101);
--local VAllImages_Nmit = torch.ByteTensor(VtotalIm_Nmit,3,101,101);
VAllImages_mit = torch.ByteTensor(VtotalIm_mit,3,101,101);
VAllImages_Nmit=torch.ByteTensor(VtotalIm_Nmit,3,101,101);
local VFoldIndex_mit = torch.Tensor(VtotalIm_mit);
local VFoldIndex_Nmit = torch.Tensor(VtotalIm_Nmit);
local Vcount_mit = 0;
local Vcount_Nmit = 0;
print("validation set:-")
for i=1,(#Folds) do
  print(i,VnumOfImages_mit[i],VnumOfImages_Nmit[i])
  local mfile = io.open('Filenames/'..Folds[i]..'_mit.txt')
  local nmfile = io.open('Filenames/'..Folds[i]..'_nmit.txt')
  local tempc=0;
  for l in io.lines('Filenames/'..Folds[i]..'_mit.txt') do
  	if tempc==VnumOfImages_mit[i] then break end
  	Vcount_mit = Vcount_mit +1;
  	tempc = tempc+1;
  	VAllImages_mit[Vcount_mit] = image.load('../Fold_Validate/'..Folds[i]..'/mitosis/'..l,3,'byte');
  	VFoldIndex_mit[Vcount_mit]=i;
  end
  tempc=0;
  for l in io.lines('Filenames/'..Folds[i]..'_nmit.txt') do
  	if tempc==VnumOfImages_Nmit[i] then break end
  	Vcount_Nmit = Vcount_Nmit +1;
  	tempc = tempc+1;
  	VAllImages_Nmit[Vcount_Nmit] = image.load('../Fold_Validate/'..Folds[i]..'/not_mitosis/'..l,3,'byte');
  	VFoldIndex_Nmit[Vcount_Nmit]=i;
  end
end
-------------------------------------------------Validate/Test set image names loaded here--------------------------------
print(numOfImages_mit)
print("all loaded!!!!")
classes = {'mitosis','non-mitosis'}

--define networks here

D1 = nn.Sequential();
D1:add(nn.SpatialConvolution(3,16,2,2));
D1:add(nn.LeakyReLU(0.2,true));
D1:add(nn.SpatialMaxPooling(2,2));
-- size: 16 X 50 X 50
D1:add(nn.SpatialConvolution(16,16,3,3));
D1:add(nn.LeakyReLU(0.2,true));
D1:add(nn.SpatialMaxPooling(2,2));
-- size: 16 X 24 X 24
D1:add(nn.SpatialConvolution(16,16,3,3));
D1:add(nn.LeakyReLU(0.2,true));
D1:add(nn.SpatialMaxPooling(2,2));
-- size: 16 X 11 X 11
D1:add(nn.SpatialConvolution(16,16,2,2));
D1:add(nn.LeakyReLU(0.2,true));
D1:add(nn.SpatialMaxPooling(2,2));
-- size: 16 X 5 X 5
D1:add(nn.SpatialConvolution(16,16,2,2));
D1:add(nn.LeakyReLU(0.2,true));
D1:add(nn.SpatialMaxPooling(2,2));
-- size: 16 X 2 X 2
D1:add(nn.View(16*2*2));
D1:add(nn.Linear(16*2*2,100));
D1:add(nn.Linear(100,2));
D1:add(nn.LogSoftMax());

--print(D1);

--print(D1:forward(trainData.data[10]))



for f=1,(#Folds) do

	local D2 = nn.Sequential();
	D2:add(nn.SpatialConvolution(3,16,4,4));
	D2:add(nn.LeakyReLU(0.2,true));
	D2:add(nn.SpatialMaxPooling(2,2));
	-- size: 16 X 49 X 49
	D2:add(nn.SpatialConvolution(16,16,4,4));
	D2:add(nn.LeakyReLU(0.2,true));
	D2:add(nn.SpatialMaxPooling(2,2));
	-- size: 16 X 23 X 23
	D2:add(nn.SpatialConvolution(16,16,4,4));
	D2:add(nn.LeakyReLU(0.2,true));
	D2:add(nn.SpatialMaxPooling(2,2));
	-- size: 16 X 10 X 10
	D2:add(nn.SpatialConvolution(16,16,3,3));
	D2:add(nn.LeakyReLU(0.2,true));
	D2:add(nn.SpatialMaxPooling(2,2));
	-- size: 16 X 4 X 4
	D2:add(nn.View(16*4*4));
	D2:add(nn.Linear(16*4*4,100));
	D2:add(nn.Linear(100,2));
	D2:add(nn.LogSoftMax());
	--print(D2);

	--cudnn.convert(D2, cudnn)
	D2:cuda()
	local criterion = nn.ClassNLLCriterion():cuda()
	local trainer = nn.StochasticGradient(D2,criterion)
	trainer.learningRate = 0.002
	trainer.learningRateDecay = 1
	trainer.shuffleIndices = 0
	trainer.maxIteration = 30
	--trainer:train(trainData)
	---------------------------------------Training parameters set----------------------------------------------------

	local trSize = 0
	local valSize = 0
	for a=1,#(Folds) do
		if ((a-f+#(Folds))%#(Folds)<5) then
			trSize = trSize + numOfImages_mit[a]+numOfImages_Nmit[a];
		else
			if ((a-f+#(Folds))%#(Folds)<10) then
				valSize = valSize + VnumOfImages_mit[a]+VnumOfImages_Nmit[a];
			end
		end
	end
	f_test = (f+10)%#(Folds);
	if f_test==0 then
		f_test = #(Folds)
	end
	local teSize = VnumOfImages_mit[f_test] + VnumOfImages_Nmit[f_test];
	print("oh")
	local trainImageAll = torch.Tensor(trSize)
	local trainLabelAll = torch.Tensor(trSize)
	local validateLabelAll = torch.Tensor(valSize)
	local testImageAll = torch.Tensor(teSize)
	local validateImageAll = torch.Tensor(valSize)
	local testLabelAll = torch.Tensor(teSize)
	--print("yeah")
	local trCount = 0;
	local teCount = 0;
	local valCount = 0;
	-- load mitosis data for all folds in validate and test for fth fold from fth to (f+4)th fold.
	for j=1,(VtotalIm_mit) do
		if ((VFoldIndex_mit[j]-f+#(Folds))%#(Folds) == 10) then
			teCount = teCount+1;
			--print(VFoldIndex_mit[j],teCount,teSize)
			testImageAll[teCount] = j;
			testLabelAll[teCount] = 1;
		else
			if ((VFoldIndex_mit[j]-f+#(Folds))%#(Folds) >=5) then
				valCount = valCount + 1
				validateImageAll[valCount] = j;
				validateLabelAll[valCount] = 1;
			end
		end
	end
	for j=1,(totalIm_mit) do
		if ((FoldIndex_mit[j]-f+#(Folds))%#(Folds) <5) then
			trCount = trCount+1;
			trainImageAll[trCount] = j;
			trainLabelAll[trCount] = 1;
		end
	end
	collectgarbage()
	print("mitosis loading done")
	print("arna",trSize,valSize,teSize)
	-- load not-mitosis data for all folds in validate and test for fth fold from fth to (f+4)th fold.
	for j=1,(VtotalIm_Nmit) do
		if ((VFoldIndex_Nmit[j]-f+#(Folds))%#(Folds) == 10) then
			teCount = teCount+1;
			testImageAll[teCount] = j;
			testLabelAll[teCount] = 2;
		else
			if ((VFoldIndex_Nmit[j]-f+#(Folds))%#(Folds) >=5) then
				valCount = valCount + 1
				validateImageAll[valCount] = j;
				validateLabelAll[valCount] = 2;
			end
		end
	end
	for j=1,(totalIm_Nmit) do
		if ((FoldIndex_Nmit[j]-f+#(Folds))%#(Folds) <5) then
			trCount = trCount+1;
			trainImageAll[trCount] = j;
			trainLabelAll[trCount] = 2;
		end
	end

	collectgarbage()
	print("non-mitosis loading done")
	
	print(trSize,valSize,teSize)
	print((#trainLabelAll)[1])


	local mean = {}
	local sum = {0,0,0}
	local std = {}
	local counter = 0;
	local trIm=torch.Tensor(3,101,101);
	for i=1,trSize do
		trIm = torch.CudaTensor(3,101,101);
		if trainLabelAll[i]==1 then
			trIm = AllImages_mit[trainImageAll[i]]:cuda();
		else
			trIm = AllImages_Nmit[trainImageAll[i]]:cuda();
		end
		for c=1,3 do
			sum[c] = sum[c]+trIm[{c,{},{}}]:sum();
		end
		counter = counter+(101*101);
	end
	for c=1,3 do
		mean[c] = sum[c]/counter;
	end
	sum = {0,0,0}
	for i=1,trSize do
		trIm = torch.CudaTensor(3,101,101);
		if trainLabelAll[i]==1 then
			trIm = AllImages_mit[trainImageAll[i]]:cuda();
		else
			trIm = AllImages_Nmit[trainImageAll[i]]:cuda();
		end
		for c=1,3 do
			trIm[{c,{},{}}]:add(-mean[c]);
			sum[c] = sum[c]+trIm[{c,{},{}}]:pow(2):sum();
		end
	end
	for c=1,3 do
		std[c] = torch.sqrt(sum[c]/counter);
	end
	print("training preprocessing done :)");

	local Vmean = {}
	local Vsum = {0,0,0}
	local Vstd = {}
	local Vcounter = 0;
	for i=1,valSize do
		valIm = torch.CudaTensor(3,101,101);
		if validateLabelAll[i]==1 then
			valIm = VAllImages_mit[validateImageAll[i]]:cuda();
		else
			valIm = VAllImages_Nmit[validateImageAll[i]]:cuda();
		end
		--valIm = image.load(validateImageAll[i]);
		for c=1,3 do
			Vsum[c] = Vsum[c]+valIm[{c,{},{}}]:sum();
		end
		Vcounter = Vcounter+(101*101);
	end
	for c=1,3 do
		Vmean[c] = Vsum[c]/Vcounter;
	end
	Vsum = {0,0,0}
	for i=1,valSize do
		valIm = torch.CudaTensor(3,101,101);
		if validateLabelAll[i]==1 then
			valIm = VAllImages_mit[validateImageAll[i]]:cuda();
		else
			valIm = VAllImages_Nmit[validateImageAll[i]]:cuda();
		end
		for c=1,3 do
			valIm[{c,{},{}}]:add(-Vmean[c]);
			Vsum[c] = Vsum[c]+valIm[{c,{},{}}]:pow(2):sum();
		end
	end
	for c=1,3 do
		Vstd[c] = torch.sqrt(Vsum[c]/Vcounter);
	end
	local ValImagesInserted = torch.Tensor(valCount):zero();
	print("validation preprocessing done :D");

	local iteration =1;
	local currentLearningRate = trainer.learningRate;
	local input=torch.CudaTensor(batchSize,3,101,101);
	local target=torch.CudaTensor(batchSize);
	local Vinput=torch.CudaTensor(3,101,101);
	local errorTensor = torch.Tensor(trainer.maxIteration);
	while true do
		local labelsShuffle = torch.randperm((#trainLabelAll)[1])
		local currentError_ = 0
        for t = 1,math.floor(trSize/batchSize) do
        	local currentError = 0;
	      	for t1 = 1,batchSize do
	      		t2 = (t-1)*batchSize+t1;
	        	target[t1] = (trainLabelAll[labelsShuffle[t2]])
	        	if target[t1]==1 then
	        		input[t1] = AllImages_mit[trainImageAll[labelsShuffle[t2]]]:cuda();
	        	else
	        		input[t1] = AllImages_Nmit[trainImageAll[labelsShuffle[t2]]]:cuda();
	        	end
				--print(t1)
	        end
	        for i=1,3 do
			   -- normalize each channel globally:
			   input[{{},i,{},{}}]:add(-mean[i])
			   input[{{},i,{},{}}]:div(std[i])
			end
	        currentError = currentError + criterion:forward(D2:forward(input), target)
			currentError_ = currentError_ + currentError
     		D2:updateGradInput(input, criterion:updateGradInput(D2:forward(input), target))
     		D2:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate)
     		print("batch "..t.." done ==>");
	    end
	    ---- training on the remaining images, i.e. left after using fixed batch size.
	    local residualInput = torch.CudaTensor(trSize%batchSize,3,101,101);
	    local residualTarget = torch.CudaTensor(trSize%batchSize);
	    for t1=1,(trSize%batchSize) do
	    	t2=batchSize*math.floor(trSize/batchSize) + t1;
	    	residualTarget[t1] = (trainLabelAll[labelsShuffle[t2]]);
	    	if residualTarget[t1]==1 then
        		residualInput[t1] = AllImages_mit[trainImageAll[labelsShuffle[t2]]]:cuda();
        	else
        		residualInput[t1] = AllImages_Nmit[trainImageAll[labelsShuffle[t2]]]:cuda();
        	end
		end
		if trSize%batchSize ~=0 then
			for i=1,3 do
			   -- normalize each channel globally:
			   residualInput[{{},i,{},{}}]:add(-mean[i])
			   residualInput[{{},i,{},{}}]:div(std[i])
			end
			currentError_ = currentError_ + criterion:forward(D2:forward(residualInput), residualTarget)
	 		D2:updateGradInput(residualInput, criterion:updateGradInput(D2:forward(residualInput), residualTarget))
	 		D2:accUpdateGradParameters(residualInput, criterion.gradInput, currentLearningRate)
	 	end

		currentError_ = currentError_ / trSize
		print("#iteration "..iteration..": current error = "..currentError_);
		errorTensor[iteration] = currentError_;
		if iteration%5==0 then
			for t=1,valSize do
				if validateLabelAll[t]==1 then
					VInput = VAllImages_mit[validateImageAll[t]]:cuda();
				else
					VInput = VAllImages_Nmit[validateImageAll[t]]:cuda();
				end
	        	Vtarget = validateLabelAll[t]
	      		for i=1,3 do
				   -- normalize each channel globally:
				   Vinput[{i,{},{}}]:add(-Vmean[i])
				   Vinput[{i,{},{}}]:div(Vstd[i])
				end
				local prediction = D2:forward(Vinput:cuda())
		    	local confidences, indices = torch.sort(prediction, true)
		    	if Vtarget~=indices[1] then
		    		if ValImagesInserted[t]==0 then
		    			tempTensor = torch.Tensor(1);
		    			tempTensor[1] = validateImageAll[t];
		    			trainImageAll = trainImageAll:cat(tempTensor);
		    			--table.insert(trainImageAll,validateImageAll[t]);
		    			tempTensor[1] = validateLabelAll[t];
		    			trainLabelAll = trainLabelAll:cat(tempTensor);
		    			--table.insert(trainLabelAll,validateLabelAll[t]);
		    			trSize = trSize+1;
		    			ValImagesInserted[t]=1;
		    		end
		    	end
		    end
		end
    	iteration = iteration + 1
      	currentLearningRate = trainer.learningRate/(1+iteration*trainer.learningRateDecay)
      	if trainer.maxIteration > 0 and iteration > trainer.maxIteration then
        	print("# StochasticGradient: you have reached the maximum number of iterations")
         	print("# training error = " .. currentError_)
         	break
      	end
	end
	print("The error curve values for fold"..f.." :-");
	print(errorTensor);

	torch.save("D2_batchFoldModel_" .. f .. ".t7",D2);
	--D2 = torch.load("D2_foldModel.t7")
	collectgarbage()



	testData = {
		data = {},
		labels = torch.Tensor(teSize),
		size = function() return teSize end
	}

	for i =1,teSize do
		testData.data[i] = testImageAll[i]
		testData.labels[i] = testLabelAll[i]
	end

	correct = 0
	class_perform = {0,0}
	class_size = {0,0}
	for i=1,teSize do
	    local groundtruth = testData.labels[i]
	    local example = torch.CudaTensor(3,101,101);
	    if groundtruth == 1 then
	    	example = VAllImages_mit[testData.data[i]];
	    else 
	    	example = VAllImages_Nmit[testData.data[i]];
	    end
	    --local example = image.load(testData.data[i])
	    for i=1,3 do
		   -- normalize each channel globally:
		   example[{ i,{},{} }]:add(-mean[i])
		   example[{ i,{},{} }]:div(std[i])
		end
	    --print('ground '..groundtruth)
	    class_size[groundtruth] = class_size[groundtruth] +1
	    local prediction = D2:forward(example:cuda())
	    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
	    if groundtruth == indices[1] then
	        correct = correct + 1
	        class_perform[groundtruth] = class_perform[groundtruth] + 1
	    end
	end
	print(sys.COLORS.green..'Fold '..f..' done. Results ---->');
	print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
	for i=1,#classes do
	   print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
	end
end
collectgarbage()
