// Gmsh project created on Mon Dec 16 09:37:16 2019
//+
lc = .1;
delta = lc;
SetFactory("OpenCASCADE");

Point(1) = {-delta,-delta,-delta,lc};
Point(2) = {-delta,-delta,1+delta,lc};
Point(3) = {-delta,1+delta,-delta,lc};
Point(4) = {-delta,1+delta,1+delta,lc};
Point(5) = {1+delta,-delta,-delta,lc};
Point(6) = {1+delta,-delta,1+delta,lc};
Point(7) = {1+delta,1+delta,-delta,lc};
Point(8) = {1+delta,1+delta,1+delta,lc};

Line(1) = {1,2};
Line(2) = {2,4};
Line(3) = {4,3};
Line(4) = {3,1};

Line(5) = {6,8};
Line(6) = {8,7};
Line(7) = {7,5};
Line(8) = {5,6};

Line(9) = {4,8};
Line(10) = {2,6};
Line(11) = {5,1};
Line(12) = {3,7};

Line Loop(1) = {2,9,-5,-10};
Line Loop(2) = {4,1,2,3};
Line Loop(3) = {11,-4,12,7};
Line Loop(4) = {6,-5,8,-7};
Line Loop(5) = {3,12,-6,-9};
Line Loop(6) = {1,-11,8,-10};

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};
Plane Surface(6) = {6};
/*
Transfinite Surface{1};
Transfinite Surface{2};
Transfinite Surface{3};
Transfinite Surface{4};
Transfinite Surface{5};
Transfinite Surface{6};
*/
Point(9) = {0,0,0,lc};
Point(10) = {0,0,1,lc};
Point(11) = {0,1,0,lc};
Point(12) = {0,1,1,lc};
Point(13) = {1,0,0,lc};
Point(14) = {1,0,1,lc};
Point(15) = {1,1,0,lc};
Point(16) = {1,1,1,lc};

Line(14) = {9,10};
Line(15) = {10,12};
Line(16) = {12,11};
Line(17) = {11,9};

Line(18) = {14,16};
Line(19) = {16,15};
Line(20) = {15,13};
Line(21) = {13,14};

Line(22) = {12,16};
Line(23) = {10,14};
Line(24) = {13,9};
Line(25) = {11,15};

Line Loop(9) = {15,22,-18,-23};
Line Loop(10) = {17,14,15,16};
Line Loop(11) = {24,-17,25,20};
Line Loop(12) = {19,-18,21,-20};
Line Loop(13) = {16,25,-19,-22};
Line Loop(14) = {14,-24,21,-23};

Plane Surface(9) = {9};
Plane Surface(10) = {10};
Plane Surface(11) = {11};
Plane Surface(12) = {12};
Plane Surface(13) = {13};
Plane Surface(14) = {14};

/*
Transfinite Surface{9};
Transfinite Surface{10};
Transfinite Surface{11};
Transfinite Surface{12};
Transfinite Surface{13};
Transfinite Surface{14};
*/

// Omega
Surface Loop(1) = {9,10,11,12,13,14};
Physical Surface(9) = {9,10,11,12,13,14};
Volume(1) = {1};
Physical Volume(1) = {1};

// Omega I
Surface Loop(2) = {1,2,3,4,5,6};
Physical Surface(99) = {1,2,3,4,5,6};
Volume(2) = {2,1};
Physical Volume(2) = {2};
