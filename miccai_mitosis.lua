require 'nn';


--[[
D1 ->
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> output]
  (1): nn.SpatialConvolution(3 -> 16, 2x2)
  (2): nn.SpatialMaxPooling(2x2, 2,2)
  (3): nn.SpatialConvolution(16 -> 16, 3x3)
  (4): nn.SpatialMaxPooling(2x2, 2,2)
  (5): nn.SpatialConvolution(16 -> 16, 3x3)
  (6): nn.SpatialMaxPooling(2x2, 2,2)
  (7): nn.SpatialConvolution(16 -> 16, 2x2)
  (8): nn.SpatialMaxPooling(2x2, 2,2)
  (9): nn.SpatialConvolution(16 -> 16, 2x2)
  (10): nn.SpatialMaxPooling(2x2, 2,2)
  (11): nn.View(64)
  (12): nn.Linear(64 -> 100)
  (13): nn.Linear(100 -> 2)
}
--]]


--input size : 3 X 101 X 101
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

print(D1);


--input size : 3 X 101 X 101
D2 = nn.Sequential();
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

print(D2);


