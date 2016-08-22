require 'nn'
trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
print(trainset)
print(#trainset.data)
print(classes[trainset.label[33]])
setmetatable(trainset,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]}
                end}
);

trainset.data = trainset.data:double()

function trainset:size()
    return self.data:size(1)
end
print(trainset:size())
print(trainset[33])

mean = {}
stdev = {}
for i=1,3 do
    mean[i] = trainset.data[{{},{i},{},{}}]:mean()
    print("Channel " .. i .. " mean: " .. mean[i])
    trainset.data[{{},{i},{},{}}]:add(-mean[i])

    stdev[i] = trainset.data[{{},{i},{},{}}]:std()
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdev[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdev[i])
end

net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(net,criterion)
trainer.learningRate = 0.001
trainer.maxIter = 10
trainer:train(trainset)


testset.data = testset.data:double()
for i=1,3 do
    testset.data[{{},{i},{},{}}]:add(-mean[i])
    testset.data[{{},{i},{},{}}]:div(stdev[i])
end

horse = testset.data[100]
print(horse:mean(), horse:std())

print(classes[testset.label[100]])
predicted = net:forward(testset.data[100])
print(predicted:exp())
for i=1,predicted:size(1) do
    print(classes[i], predicted[i])
end

correct = 0
class_perform = {0,0,0,0,0,0,0,0,0,0}
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
        class_perform[groundtruth] = class_perform[groundtruth] + 1
    end
end

print("Overall correct " .. correct .. " percentage correct" .. (100*correct/10000) .. " % ")
for i=1,#classes do
	print(classes[i], 100*class_perform[i]/1000 .. " % ")
end




